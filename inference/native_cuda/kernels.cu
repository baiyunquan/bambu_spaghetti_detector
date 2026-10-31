// CUDA 10.2 / Maxwell sm53. No framework, CUB, or host compiler dependency.
struct Candidate {
    float x1, y1, x2, y2, score;
    int cls, anchor;
};

// CPU-compatible 8-bit bilinear resize. Half-pixel coordinates and 11-bit
// coefficients are prepared from dimensions only; every image pixel stays GPU.
// Integer rounding follows OpenCV's INTER_LINEAR 8-bit arithmetic, rather than
// cuda::resize's different corner-aligned sampling convention.
extern "C" __global__ void resize_compatible(
    const unsigned char* src, unsigned long long src_pitch, int sw, int sh,
    unsigned char* dst, unsigned long long dst_pitch, int dw, int dh,
    const int* horizontal, const int* vertical)
{
    int item=blockIdx.x*blockDim.x+threadIdx.x;
    if (item>=dw*dh*3) return;
    int c=item%3, x=(item/3)%dw, y=item/(dw*3);
    int sx=horizontal[x*3], sy=vertical[y*3];
    int a=horizontal[x*3+1], b=horizontal[x*3+2];
    int u=vertical[y*3+1], v=vertical[y*3+2];
    int sx1=min(sx+1,sw-1);
    int sy0=max(0,min(sy,sh-1)), sy1=max(0,min(sy+1,sh-1));
    int top=src[sy0*src_pitch+sx*3+c]*a+src[sy0*src_pitch+sx1*3+c]*b;
    int bottom=src[sy1*src_pitch+sx*3+c]*a+src[sy1*src_pitch+sx1*3+c]*b;
    int value=(((u*(top>>4))>>16)+((v*(bottom>>4))>>16)+2)>>2;
    dst[y*dst_pitch+x*3+c]=(unsigned char)value;
}

extern "C" __global__ void decode(
    const float* boxes, const float* scores, Candidate* out, int* counters,
    int n, int classes, int model, float threshold, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x, y, w, h, score;
    int cls = 0;
    bool valid = true;
    if (model == 0) {
        x=boxes[4*i]; y=boxes[4*i+1]; w=boxes[4*i+2]; h=boxes[4*i+3];
        score=scores[i];
        x*=width; w*=width; y*=height; h*=height;
    } else {
        int row = i * (5 + classes);
        x=model==1 ? boxes[i] : boxes[row];
        y=model==1 ? boxes[n+i] : boxes[row+1];
        w=model==1 ? boxes[2*n+i] : boxes[row+2];
        h=model==1 ? boxes[3*n+i] : boxes[row+3];
        float obj = model==1 ? 1.f : boxes[row+4];
        valid = isfinite(obj);
        score=-__int_as_float(0x7f800000);
        for (int c=0; c<classes; ++c) {
            float p = model==1 ? boxes[(4+c)*n+i] : boxes[row+5+c];
            valid = valid && isfinite(p);
            float value = p*obj;
            valid = valid && isfinite(value);
            if (value>score) { score=value; cls=c; }
        }
        float cx=x, cy=y;
        x=cx-w*.5f; y=cy-h*.5f;
        w=cx+w*.5f; h=cy+h*.5f;
    }
    valid = valid && isfinite(x) && isfinite(y) && isfinite(w) &&
            isfinite(h) && isfinite(score);
    if (!valid) { atomicExch(counters+1,1); return; }
    if (score < threshold) return;
    int slot=atomicAdd(counters,1);
    Candidate c={x,y,w,h,score,cls,i};
    out[slot]=c;
}

extern "C" __global__ void pad(Candidate* data, int count, int padded) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count && i<padded) {
        Candidate c={0,0,0,0,-__int_as_float(0x7f800000),-1,2147483647};
        data[i]=c;
    }
}

__device__ bool before(Candidate a, Candidate b) {
    return a.score>b.score || (a.score==b.score && a.anchor<b.anchor);
}
extern "C" __global__ void bitonic(Candidate* data, int n, int j, int k) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int other=i^j;
    if (i>=n || other>=n || other<=i) return;
    Candidate a=data[i], b=data[other];
    bool descending=(i&k)==0;
    if ((descending && before(b,a)) || (!descending && before(a,b))) {
        data[i]=b; data[other]=a;
    }
}

__device__ float iou(Candidate a, Candidate b) {
    float w=fmaxf(0.f,fminf(a.x2,b.x2)-fmaxf(a.x1,b.x1));
    float h=fmaxf(0.f,fminf(a.y2,b.y2)-fmaxf(a.y1,b.y1));
    float intersection=w*h;
    float total=fmaxf(0.f,a.x2-a.x1)*fmaxf(0.f,a.y2-a.y1)+
                fmaxf(0.f,b.x2-b.x1)*fmaxf(0.f,b.y2-b.y1)-intersection;
    return total>0.f ? intersection/total : 0.f;
}

extern "C" __global__ void overlap_mask(
    const Candidate* data, unsigned long long* mask,
    int n, int stride, float threshold)
{
    int rowBlock=blockIdx.y, colBlock=blockIdx.x;
    if (rowBlock>colBlock) return;
    __shared__ Candidate column[64];
    int tid=threadIdx.x, col=colBlock*64+tid;
    if (col<n) column[tid]=data[col];
    __syncthreads();
    int row=rowBlock*64+tid;
    if (row>=n) return;
    Candidate a=data[row];
    int end=min(64,n-colBlock*64);
    int start=rowBlock==colBlock ? tid+1 : 0;
    unsigned long long bits=0;
    for (int j=start;j<end;++j) {
        Candidate b=column[j];
        if (a.cls==b.cls && iou(a,b)>threshold) bits |= 1ULL<<j;
    }
    mask[row*stride+colBlock]=bits;
}

extern "C" __global__ void select_indices(
    const unsigned long long* mask, int* keep, int* count, int n, int stride)
{
    // ceil(32768/64)=512 words; engine capacity is checked before allocation.
    __shared__ unsigned long long removed[512];
    __shared__ int accepted, take;
    int tid=threadIdx.x, words=(n+63)/64;
    for (int j=tid;j<words;j+=blockDim.x) removed[j]=0;
    if (tid==0) accepted=0;
    __syncthreads();
    for (int i=0;i<n;++i) {
        if (tid==0) {
            take=(removed[i/64] & (1ULL<<(i%64)))==0;
            if (take) keep[accepted++]=i;
        }
        __syncthreads();
        if (take) {
            for (int j=i/64+tid;j<words;j+=blockDim.x)
                removed[j] |= mask[i*stride+j];
        }
        __syncthreads();
    }
    if (tid==0) *count=accepted;
}

extern "C" __global__ void gather(
    const Candidate* data, const int* keep, Candidate* output, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) output[i]=data[keep[i]];
}
