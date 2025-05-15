This is a simple cuda path tracer which renders the famous cornell box. Tested on Google colab using GPU.

compile command:
!nvcc cuda_path_tracer_cornell_box.cu -o cuda_path_tracer_cornell_box \
    -gencode arch=compute_75,code=sm_75

Testing: To test this, run the below command with samplePerPixel as the argument.
!./cuda_path_tracer_cornell_box 1000

![cornell_box_1000_spp](https://github.com/Sandip-Kumar-Shukla/Cuda_Path_Tracer/blob/main/output-3.ppm)







