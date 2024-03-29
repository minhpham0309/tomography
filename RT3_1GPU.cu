#include <iostream>
#include "mex.h"
#include "omp.h"
#include <cmath>
#include "gpu/mxGPUArray.h"
//#include "splinterp.h"
using namespace std;

#if __CUDA_ARCH__ < 600
template <typename T>
__device__ double atomicAdd(T* address, T val)
{
    unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
        __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif


template <typename T>
void __global__ updateRec(T*rec_new, int dimx, int dimy){
    //long long const i = blockDim.x * blockIdx.x + threadIdx.x;
    int const x = blockIdx.x * blockDim.x + threadIdx.x;;
    int const y = blockIdx.y;
    int const z = blockIdx.z;
    if (x<dimx ) {
        long long i = (long long)z*dimx*dimy + y*dimx +  x;
        rec_new[i] = max( 0.0, rec_new[i] );
    }
}

template <typename T>
void __global__ compute_xy_shift( const T*Matrix, const T* shift,  T*x_shift, T*y_shift, int Num_pjs){
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<Num_pjs ) {
        int index = 9*i;
        for (int j=0; j<4; j++){
            x_shift[4*i+j] = Matrix[index+0]*shift[2*j] + Matrix[index+3]*0.0 + Matrix[index+6]*shift[2*j+1] ;
            y_shift[4*i+j] = Matrix[index+1]*shift[2*j] + Matrix[index+4]*0.0 + Matrix[index+7]*shift[2*j+1] ;
        }
    }   
    
}

template <typename T>
void __global__ R1norm(const T *d_vec, double* R1, int N){
    long long const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N ) {        
        atomicAdd( R1 , (double)abs(d_vec[i]) );
    }
}

template <typename T>
void __global__ R1norm(const T *d_vec, T* R1, int N){
    long long const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N ) {        
        atomicAdd( R1 , abs(d_vec[i]) );
    }
}

template <typename T>
void __global__ computeCoords(T* coords, const int dimx, const int dimy, const int ncx, const int ncy, int ncz, long long N, long long starting_point){
    long long i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N ) { 
        i+=starting_point;
        coords[3*i]   =  int(           i%dimx ) - ncx  + 1 ;
        //coords[3*i+1] =  int(  ( i%(dimx*dimy) ) /dimx ) - ncy + 1;
        coords[3*i+1] =  ( int( i/dimx ) ) % dimy - ncy + 1;
        coords[3*i+2] =  int(    i/(dimx*dimy) ) - ncz + 1 ;
    }    
}
//xx_h = mod( ((0:692)'),7)-3;
//yy_h = floor( mod( (0:692)', 7*9) /7)-4; yy_h = mod( floor( (0:692)' /7), 9 )-4;
//zz_h = floor( (0:692)'/(7*9)) - 5;

template <typename T>
void __global__ setValue(T*residual, double val, int N){
    long long const i = blockDim.x * blockIdx.x + threadIdx.x;
    //T o_ratio_inv = 1.0/o_ratio;
    if (i<N) {
        residual[i] = val;
    }
}

static const int blockSize = 1024;
static const int gridSize = 24; 
template <typename T>
__global__ void sumCommMultiBlock(const T *gArr, long long arraySize, T *gOut) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    T sum = 0;
    for (int i = gthIdx; i < arraySize; i += gridSize)
        sum += abs(gArr[i]);
    __shared__ T shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { //uniform
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}

template <typename T>
__host__ double sumArray(const T* arr, long long wholeArraySize) {
    T* arr_pt;
    cudaMalloc((void**)&arr_pt, wholeArraySize * sizeof(T));
    cudaMemcpy(arr_pt, arr, wholeArraySize * sizeof(T), cudaMemcpyHostToDevice);

    T sum;
    T* sum_pt;
    cudaMalloc((void**)&sum_pt, sizeof(T)*gridSize);
    
    sumCommMultiBlock<<<gridSize, blockSize>>>(arr_pt, wholeArraySize, sum_pt);
    //dev_out now holds the partial result
    sumCommMultiBlock<<<1, blockSize>>>(sum_pt, gridSize, sum_pt);
    //dev_out[0] now holds the final result
    cudaDeviceSynchronize();
    
    cudaMemcpy(&sum, sum_pt, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(arr_pt);
    cudaFree(sum_pt);
    return sum;
}




template <typename T>
void __global__ computeResidual(T*residual, const T* projections, const T scale, long long N){
    long long const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<N) {
        residual[i] = residual[i]*scale - projections[i];
    }
}

template <typename T>
void __global__ RadonTF(const T*data,const T*Matrix, const int nrows, const int ncols,const T*nc, const int o_ratio,const T*x_s,const T*y_s,
T*result, int dimz){
    //long long const i = blockDim.x * blockIdx.x + threadIdx.x;    
    int const x = blockIdx.x * blockDim.x + threadIdx.x;;
    int const y = blockIdx.y;
    int const z = blockIdx.z;
    
    int origin_offset = 1;
    long s ;  
    //#pragma omp parallel for default(shared) private(i,s) schedule(static)
    if (x<nrows){
        const T & data_i = data[z*ncols*nrows + y*nrows + x];
        //const T coord_x = int(           i%nrows ) - nc[0]  + 1 ;
        //const T coord_y = ( int( i/nrows ) ) % ncols - nc[1] + 1;
        const T coord_x = x - nc[0] + 1;;
        const T coord_y = y - nc[1] + 1;
        const T coord_z = z - nc[2] + 1 ;

        //long index = i*3;
        const T x_i = Matrix[0]*coord_x + Matrix[3]*coord_y + Matrix[6]*coord_z + nc[0];
        const T y_i = Matrix[1]*coord_x + Matrix[4]*coord_y + Matrix[7]*coord_z + nc[1];

        for (s=0; s<o_ratio; s++){
            //for (i = 0; i < N; ++i) {
            T x_is = x_i + x_s[s] - origin_offset;
            T y_is = y_i + y_s[s] - origin_offset;

            // get coordinates of bounding grid locations
            long long x_1 = ( long long) floor(x_is) ;
            long long x_2 = x_1 + 1;
            long long y_1 = ( long long) floor(y_is) ;
            long long y_2 = y_1 + 1;
            
            if (x_1>=-1 && x_2<=nrows  &&  y_1>=-1 && y_2<=ncols ){ 
                T w_x1 = x_2 - x_is ;
                T w_x2 = 1   - w_x1;
                T w_y1 = y_2 - y_is ;
                T w_y2 = 1   - w_y1;            
                if (x_1==-1){
                    if (y_1==-1){
                        atomicAdd( &result[x_2 + y_2*nrows] , w_x2*w_y2 * data_i);
                    }
                    else if(y_2==ncols){
                        atomicAdd( &result[x_2 + y_1*nrows] , w_x2*w_y1 * data_i);
                    }
                    else{
                        atomicAdd( &result[x_2 + y_1*nrows] , w_x2*w_y1 * data_i);
                        atomicAdd( &result[x_2 + y_2*nrows] , w_x2*w_y2 * data_i);                    
                    }
                }
                else if (x_2==nrows){
                    if (y_1==-1){
                        atomicAdd( &result[x_1 + y_2*nrows] , w_x1*w_y2 * data_i);
                    }
                    else if(y_2==ncols){
                        atomicAdd( &result[x_1 + y_1*nrows] , w_x1*w_y1 * data_i);
                    }
                    else{
                        atomicAdd( &result[x_1 + y_1*nrows] , w_x1*w_y1 * data_i);
                        atomicAdd( &result[x_1 + y_2*nrows] , w_x1*w_y2 * data_i);                  
                    } 
                }
                else{
                    if (y_1==-1){
                        atomicAdd( &result[x_1 + y_2*nrows] , w_x1*w_y2 * data_i);
                        atomicAdd( &result[x_2 + y_2*nrows] , w_x2*w_y2 * data_i);
                    }
                    else if(y_2==ncols){
                        atomicAdd( &result[x_1 + y_1*nrows] , w_x1*w_y1 * data_i);
                        atomicAdd( &result[x_2 + y_1*nrows] , w_x2*w_y1 * data_i);
                    }
                    else{
                        atomicAdd( &result[x_1 + y_1*nrows] , w_x1*w_y1 * data_i);
                        atomicAdd( &result[x_1 + y_2*nrows] , w_x1*w_y2 * data_i);
                        atomicAdd( &result[x_2 + y_1*nrows] , w_x2*w_y1 * data_i);
                        atomicAdd( &result[x_2 + y_2*nrows] , w_x2*w_y2 * data_i);                  
                    }                               
                }
            }
        }
    }
}


template <typename T>
void __global__ RadonTpose_updateRec(const T* Matrix, const int nrows, const int ncols, const T* nc, const T* data, 
const int o_ratio, const T*x_s,const T*y_s, T* Rec, float dt, long long N){
    //long long const i = blockDim.x * blockIdx.x + threadIdx.x;
    int const x = blockIdx.x * blockDim.x + threadIdx.x;
    int const y = blockIdx.y;
    int const z = blockIdx.z;
    int origin_offset = 1;
    long s;
    //#pragma omp parallel for default(shared) private(s) schedule(static)  
    if( x < nrows ) {
        //const T coord_x = int(           i%nrows )   - nc[0] + 1 ;
        //const T coord_y = ( int( i/nrows ) ) % ncols - nc[1] + 1;
        //const T coord_z =  int(    i/(nrows*ncols) ) - nc[2] + 1 ;
        const T coord_x = x - nc[0] + 1;;
        const T coord_y = y - nc[1] + 1;
        const T coord_z = z - nc[2] + 1 ;
        long long i = (long long)z*ncols*nrows + y*nrows + x;

        //long index = i*3;
        const T x0 = Matrix[0]*coord_x + Matrix[3]*coord_y + Matrix[6]*coord_z + nc[0];
        const T y0 = Matrix[1]*coord_x + Matrix[4]*coord_y + Matrix[7]*coord_z + nc[1];
        for (s=0; s<o_ratio; s++){
            T x_i = x0 + x_s[s];
            T y_i = y0 + y_s[s];
            // get coordinates of bounding grid locations
            long long x_1 = ( long long) floor(x_i) - origin_offset;
            long long x_2 = x_1 + 1;
            long long y_1 = ( long long) floor(y_i) - origin_offset;
            long long y_2 = y_1 + 1;
            
            // handle special case where x/y is the last element
            if ( (x_i - origin_offset) == (nrows-1) )   { x_2 -= 1; x_1 -= 1;}
            if ( (y_i - origin_offset) == (ncols-1) )   { y_2 -= 1; y_1 -= 1;}
            
            // return 0 for target values that are out of bounds
            if (x_1 < 0 | x_2 > (nrows - 1) |  y_1 < 0 | y_2 > (ncols - 1)){
                //result[i] = 0;
            }
            else {
                // get the array values
                const T& f_11 = data[x_1 + y_1*nrows];
                const T& f_12 = data[x_1 + y_2*nrows];
                const T& f_21 = data[x_2 + y_1*nrows];
                const T& f_22 = data[x_2 + y_2*nrows];
                
                // compute weights
                T w_x1 = x_2 - (x_i - origin_offset);
                T w_x2 = (x_i - origin_offset) - x_1;
                T w_y1 = y_2 - (y_i - origin_offset);
                T w_y2 = (y_i - origin_offset) - y_1;
                
                T a,b;
                a = f_11 * w_x1 + f_21 * w_x2;
                b = f_12 * w_x1 + f_22 * w_x2;
                Rec[i] -= dt*(a * w_y1 + b * w_y2);
            }
        }
    }
}


void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
    int const threadsPerBlock = 256;
    int blocksPerGridPrj, blocksPerGridRec;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();    
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(0);            // Set device 0 as current    
    if(err!=cudaSuccess){
        printf("cuda fail to set\n");
    }

    /*
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
        device, deviceProp.major, deviceProp.minor);
    }
    */

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";    
    if ((nrhs!=6)&&(nrhs!=7) ){ //|| (nlhs!=1)  ) {
        cout << "error0"<<endl;
        mexErrMsgIdAndTxt(errId, errMsg); //!(mxIsGPUArray(prhs[0]))
    }
    /*
    * 0: projections
    * 1: mat
    * 2: coord
    * 3: nc
    * 4: rec
    * 5: x_shift3
    * 6: y_shift3
    * 7: z_shift3
    */

    float const * projections   = mxGetSingles(prhs[0]);
    float const * Matrix        = mxGetSingles(prhs[1]);
    //mxGPUArray const *Coord   = mxGPUCreateFromMxArray(prhs[2]);
    //mxGPUArray const *nc      = mxGPUCreateFromMxArray(prhs[3]);    
    //mxGPUArray const *xs2     = mxGPUCreateFromMxArray(prhs[3]);
    //mxGPUArray const *ys2     = mxGPUCreateFromMxArray(prhs[4]);   
    const double * dimzPtr      = mxGetPr(prhs[2]);  int dimz          = int( dimzPtr[0] ); 
    const double * iterationPtr = mxGetPr(prhs[3]);  int iterations    = int( iterationPtr[0] ); 
    const double * step_sizePtr = mxGetPr(prhs[4]);  double step_size  = double( step_sizePtr[0] ); 
    const bool   * positivePtr  = mxGetLogicals(prhs[5]); bool positivity = bool( positivePtr[0] );     

    const size_t o_ratio=4;
    const mwSize* projsSize   = (mxGetDimensions(prhs[0]));
    const mwSize dimx    = projsSize[0];
    const mwSize dimy    = projsSize[1];
    const mwSize Num_pjs = projsSize[2];

    const long long nrow_cols  = dimx*dimy;
    const long long nPjsPoints = dimx*dimy*Num_pjs;
    const long long recPoints = dimx*dimy*dimz;
    const mwSize recSize[] = {dimx,dimy,dimz};
    //cout << dimx <<", " << dimy <<", " << dimz <<", "  << endl;

    const mwSize* dims_Mat = (mxGetDimensions(prhs[1]));
    const mwSize Num_pjs2   = dims_Mat[2];
    const mwSize R1 = dims_Mat[0];
    const mwSize R2 = dims_Mat[1];

    /*
    const mwSize* o_ratio_dim = mxGPUGetDimensions(xs2);
    const size_t o_ratio      = o_ratio_dim[0];
    const size_t Num_pjs2      = o_ratio_dim[1];                
    */
    if(Num_pjs2!=Num_pjs || R1!=3 || R2!=3 ){
        cout << "error"<<endl;
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    if(mxGetClassID(prhs[0]) == mxDOUBLE_CLASS || mxGetClassID(prhs[1]) == mxDOUBLE_CLASS ){     
        printf("can't work with double\n");
        return;
    }    

    const mwSize ncx = int (floor(dimx/2.0)+1);
    const mwSize ncy = int (floor(dimy/2.0)+1);
    const mwSize ncz = int (floor(dimz/2.0)+1);

    const double dt =  (step_size/Num_pjs/dimz/o_ratio);
    //mexPrintf("%d\n",npoints);   
    
    // copy projections to GPU
    float * d_projections;
    cudaMalloc( &d_projections, nPjsPoints*sizeof(float) );
    cudaMemcpy(  d_projections, projections, nPjsPoints*sizeof(float), cudaMemcpyHostToDevice );

    // copy rotation matrix to GPU
    float * d_Matrix;
    cudaMalloc( &d_Matrix, 9*Num_pjs*sizeof(float) );
    cudaMemcpy( d_Matrix, Matrix, 9*Num_pjs*sizeof(float), cudaMemcpyHostToDevice );

    // plhs[0] is the returning reconstruction
    plhs[0] = mxCreateNumericArray(3, recSize, mxSINGLE_CLASS, mxREAL);
    float * rec   = (float*)mxGetSingles(plhs[0]);

    // create reconstruciton on GPU
    float * d_rec;
    cudaMalloc( &d_rec, recPoints*sizeof(float) );

    if(nrhs==7){
        const mwSize* recSize2   = mxGetDimensions(prhs[6]);
        const mwSize dimx2   = recSize2[0];
        const mwSize dimy2   = recSize2[1];
        const mwSize dimz2   = recSize2[2];
        if( dimx2!=dimx || dimy2!=dimy || dimz2!=dimz ){
            cout << "error2"<<endl;
            mexErrMsgIdAndTxt(errId, errMsg);
        }
        float * rec_ori = (float*)mxGetSingles(prhs[6]);        
        //memcpy( h_rec, rec_ori, recPoints*sizeof(float) );
        cudaMemcpy( d_rec, rec_ori, recPoints*sizeof(float), cudaMemcpyHostToDevice);
    }     
    else{
        cudaMemset(d_rec, 0, recPoints*sizeof(float));
    }
    
    // create residual on GPU
    float * d_residual;
    cudaMalloc( &d_residual, nPjsPoints*sizeof(float) );
    /*
    mxGPUArray * residual = mxGPUCreateGPUArray(mwSize(3),
                        pjsSize, mxGPUGetClassID(projections),
                        mxREAL, MX_GPU_INITIALIZE_VALUES );
    */
    
    
    //blocksPerGrid    = (recPoints + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGridPrj = (nPjsPoints + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGridRec = (recPoints + threadsPerBlock - 1) / threadsPerBlock;
    dim3 blocksPerGridRec2( (dimx + threadsPerBlock - 1) / threadsPerBlock , dimy, dimz );

    //mxGPUArray const *support     = mxGPUCreateFromMxArray(prhs[7]);
    //bool const *d_support = (bool const *) (mxGPUGetDataReadOnly(support));            


    //float const *d_projections = (float const *) (mxGPUGetDataReadOnly(projections));
    //float const *d_Matrix      = (float const *) (mxGPUGetDataReadOnly(Matrix));
    //float const *d_Coord       = (float const *) (mxGPUGetDataReadOnly(Coord));
    //float const *d_xs         = (float const *) (mxGPUGetDataReadOnly(xs2));
    //float const *d_ys         = (float const *) (mxGPUGetDataReadOnly(ys2));
    //float *d_residual = (float *) (mxGPUGetData(residual));


    // compute rotated shift
    float shift[]  = {0.25,0.25, 0.25,-0.25,-0.25,0.25,-0.25,-0.25};
    float *shift_ptr;    
    cudaMalloc( (void**) &shift_ptr, 8*sizeof(float) );
    cudaMemcpy( shift_ptr, shift, 8*sizeof(float), cudaMemcpyHostToDevice);
    float * x_shift, *y_shift;
    cudaMalloc( (void**) &x_shift, 4*Num_pjs*sizeof(float) );
    cudaMalloc( (void**) &y_shift, 4*Num_pjs*sizeof(float) );
    compute_xy_shift<<<2, threadsPerBlock>>>( d_Matrix, shift_ptr, x_shift, y_shift, Num_pjs );
    float const *d_xs2         = (float  *) x_shift;
    float const *d_ys2         = (float  *) y_shift;

    // compute cartesian coordinates
    //computeCoords<<<blocksPerGridRec,threadsPerBlock>>>(d_Coord, dimx, dimy, ncx,ncy, ncz, recPoints, 0);

    // compute nc = [ncx,ncy,ncz]
    //const float nc_cpu[]  = { ncx,ncy,ncz}; 
    const float nc_cpu[]  = { float(floor(dimx/2.0)+1), float(floor(dimy/2.0)+1), float(floor(dimz/2.0)+1)};     
    float * d_nc;
    cudaMalloc( (void**)&d_nc, 3*sizeof(float) );
    cudaMemcpy( d_nc, nc_cpu, 3*sizeof(float), cudaMemcpyHostToDevice ); 

    // compute norm of projection
    /*double pj_norm,  *pj_norm_pt;
    cudaMalloc( (void**)&pj_norm_pt, sizeof(double) );
    cudaMemset(pj_norm_pt, 0, 1*sizeof(double));   //setValue<<<1,1>>>(pj_pt, 0.0, 1);
    R1norm<<<blocksPerGridPrj, threadsPerBlock>>>(d_projections,  pj_norm_pt, nPjsPoints);
    cudaMemcpy( &pj_norm ,  pj_norm_pt, sizeof(double), cudaMemcpyDeviceToHost ) ;
    cudaFree( pj_norm_pt);
    */
    
    double pj_norm = sumArray(d_projections,nPjsPoints);

    // set zero to residual
    //setValue<<<blocksPerGridPrj, threadsPerBlock>>>(d_residual, 0.0, nPjsPoints);        

    // iteration
    for (long iter=0; iter<iterations; iter++){
        // set zero residual
        //setValue<<<blocksPerGridPrj, threadsPerBlock>>>(d_residual, 0.0, nPjsPoints);
        cudaMemset( d_residual, 0, nPjsPoints*sizeof(float) );

        // compute forward projection        
        for (int i = 0; i < Num_pjs; i++){
            RadonTF<<<blocksPerGridRec2, threadsPerBlock>>>(d_rec, d_Matrix + i*9, dimx,dimy,d_nc,
            o_ratio,d_xs2+i*o_ratio,d_ys2+i*o_ratio, d_residual + i*nrow_cols, dimz);
        }

        // compute residual: = forward projection - measure projections
        computeResidual<<<blocksPerGridPrj, threadsPerBlock>>>(d_residual,d_projections,1.0f/o_ratio, nPjsPoints);

        // compute R1 factor
        if (iter%2 == 1 ){
            /*
            double res_norm, *res_norm_pt;
            cudaMalloc( (void**)&res_norm_pt, sizeof(double) );                
            cudaMemset(res_norm_pt, 0, 1*sizeof(double)); //setValue<<<1,1>>>(res_norm_pt, 0.0, 1);
            R1norm<<<blocksPerGridPrj, threadsPerBlock>>>(d_residual,    res_norm_pt, nPjsPoints);
            cudaMemcpy( &res_norm, res_norm_pt, sizeof(double), cudaMemcpyDeviceToHost ) ;
            cudaFree( res_norm_pt );  
            */
            float res_norm, *sum_pt;
            cudaMalloc((void**)&sum_pt, sizeof(float)*gridSize);
            sumCommMultiBlock<<<gridSize, blockSize>>>(d_residual, nPjsPoints, sum_pt);
            sumCommMultiBlock<<<1, blockSize>>>(sum_pt, gridSize, sum_pt);
            cudaDeviceSynchronize();
            cudaMemcpy(&res_norm, sum_pt, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(sum_pt);

            cout << iter+1 << ". R1 = " << res_norm/pj_norm << endl;
        }            

        // back projection
        for (long i = 0; i < Num_pjs; i ++){
            RadonTpose_updateRec<<<blocksPerGridRec2, threadsPerBlock>>>(d_Matrix + i*9, dimx,dimy, d_nc, d_residual + i*nrow_cols,
            o_ratio, d_xs2+i*o_ratio, d_ys2+i*o_ratio, d_rec, dt, recPoints);
        }

        // update rec
        if(positivity){
            updateRec<<<blocksPerGridRec2, threadsPerBlock>>>(d_rec, dimx, dimy);
        }
    }

    /* return result  */
    //plhs[0] = mxGPUCreateMxArrayOnCPU(rec);
    cudaMemcpy( rec, d_rec, recPoints*sizeof(float), cudaMemcpyDeviceToHost);
    //plhs[1] = mxGPUCreateMxArrayOnGPU(residual);

    //mxGPUDestroyGPUArray(projections); 
    //mxGPUDestroyGPUArray(rec);        
    cudaFree(d_rec);
    cudaFree(d_residual);
    cudaFree(d_Matrix);
    cudaFree(d_projections);
    
    //mxGPUDestroyGPUArray(Matrix);
    //mxGPUDestroyGPUArray(nc);
    cudaFree(d_nc);
    //mxGPUDestroyGPUArray(xs2);
    //mxGPUDestroyGPUArray(ys2);
    //mxGPUDestroyGPUArray(residual);
    //mxGPUDestroyGPUArray(support); 
    cudaFree( shift_ptr );
    cudaFree( x_shift );
    cudaFree( y_shift );
    
    //cudaDeviceReset();      
    //mxGPUArray       *coord2 = (mxGPUArray*)d_Coord2;
    // plhs[1] = mxGPUCreateMxArrayOnCPU(coord2);
    
}


/*
// compute diffnorm of xyshift
float x_norm, x_norm2,  *x_pt, *x_pt2;
cudaMalloc( (void**)&x_pt, sizeof(float) );
cudaMalloc( (void**)&x_pt2, sizeof(float) );
cudaMemset(x_pt, 0, sizeof(float));
cudaMemset(x_pt2, 0, sizeof(float));
R1norm<<<2, threadsPerBlock>>>(d_xs,  x_pt, Num_pjs*4);
R1norm<<<2, threadsPerBlock>>>(d_xs2,  x_pt2, Num_pjs*4);
cudaMemcpy( &x_norm ,  x_pt, sizeof(float), cudaMemcpyDeviceToHost ) ;
cudaMemcpy( &x_norm2 ,  x_pt2, sizeof(float), cudaMemcpyDeviceToHost ) ;
cudaFree( x_pt);
cout << "x_norm = " << x_norm << ", x_norm2 = " << x_norm2 <<endl;
*/



















