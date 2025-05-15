#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Vector math operators
__device__ __host__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __host__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ __host__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ __host__ float3 operator*(float a, const float3& b) {
    return make_float3(a*b.x, a*b.y, a*b.z);
}

__device__ __host__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ __host__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device__ __host__ float3 operator/(const float3& a, float b) {
    return make_float3(a.x/b, a.y/b, a.z/b);
}

__device__ __host__ float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __host__ float dot(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ __host__ float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z,
                       a.x*b.y - a.y*b.x);
}

__device__ __host__ float3 normalize(const float3& v) {
    float len = sqrtf(dot(v, v));
    return len > 0 ? v * (1.0f/len) : make_float3(0,0,0);
}

struct Material {
    enum Type { DIFFUSE, EMISSIVE };
    Type type;
    float3 albedo;
    float3 emission;
};

struct Quad {
    float a;
    int axis;
    float min1, max1;
    float min2, max2;
    float3 normal;
    Material material;
};

struct Ray {
    float3 origin;
    float3 direction;
};

struct HitRecord {
    float t;
    float3 point;
    float3 normal;
    Material material;
    bool front_face;

    __device__ void set_face_normal(const Ray& ray, const float3& outward_normal) {
        front_face = dot(ray.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

__device__ bool hit_quad(const Ray& ray, const Quad& quad, float t_min, float t_max, HitRecord& rec) {
    float t;
    float3 origin = ray.origin;
    float3 dir = ray.direction;

    if (quad.axis == 0) {
        if (fabs(dir.x) < 1e-8) return false;
        t = (quad.a - origin.x) / dir.x;
        float y = origin.y + t * dir.y;
        float z = origin.z + t * dir.z;
        if (t < t_min || t > t_max || y < quad.min1 || y > quad.max1 || z < quad.min2 || z > quad.max2)
            return false;
    } 
    else if (quad.axis == 1) {
        if (fabs(dir.y) < 1e-8) return false;
        t = (quad.a - origin.y) / dir.y;
        float x = origin.x + t * dir.x;
        float z = origin.z + t * dir.z;
        if (t < t_min || t > t_max || x < quad.min1 || x > quad.max1 || z < quad.min2 || z > quad.max2)
            return false;
    } 
    else {
        if (fabs(dir.z) < 1e-8) return false;
        t = (quad.a - origin.z) / dir.z;
        float x = origin.x + t * dir.x;
        float y = origin.y + t * dir.y;
        if (t < t_min || t > t_max || x < quad.min1 || x > quad.max1 || y < quad.min2 || y > quad.max2)
            return false;
    }

    rec.t = t;
    rec.point = origin + dir * t;
    rec.normal = quad.normal;
    rec.material = quad.material;
    rec.set_face_normal(ray, quad.normal);
    return true;
}

__device__ bool hit_scene(const Ray& ray, Quad* scene, int num_quads, float t_min, float t_max, HitRecord& rec) {
    HitRecord temp_rec;
    bool hit = false;
    float closest = t_max;

    for (int i = 0; i < num_quads; ++i) {
        if (hit_quad(ray, scene[i], t_min, closest, temp_rec)) {
            hit = true;
            closest = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit;
}

__device__ float3 random_in_hemisphere(curandState* rand_state, const float3& normal) {
    float3 dir;
    do {
        dir = 2.0f * make_float3(curand_uniform(rand_state), 
                                curand_uniform(rand_state),
                                curand_uniform(rand_state)) - make_float3(1,1,1);
    } while (dot(dir, dir) >= 1.0f || dot(dir, normal) < 0);
    return normalize(dir);
}

__device__ float3 trace_ray(Ray ray, Quad* scene, int num_quads, curandState* rand_state) {
    float3 throughput = make_float3(1,1,1);
    float3 color = make_float3(0,0,0);
    int depth = 0;
    int max_depth = 8;

    while (depth++ < max_depth) {
        HitRecord rec;
        if (!hit_scene(ray, scene, num_quads, 0.001f, 1e8f, rec)) break;

        if (rec.material.type == Material::EMISSIVE) {
            color += throughput * rec.material.emission;
            break;
        }

        float3 target = rec.point + rec.normal + random_in_hemisphere(rand_state, rec.normal);
        ray.origin = rec.point;
        ray.direction = normalize(target - rec.point);
        throughput = throughput * rec.material.albedo;
    }
    return color;
}

__global__ void render_kernel(float3* fb, int w, int h, int spp, Quad* scene, int num_quads) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= w || row >= h) return;

    curandState rand_state;
    curand_init(clock64(), col + row * w, 0, &rand_state);

    float3 lookfrom = make_float3(278, 278, -800);
    float3 lookat = make_float3(278, 278, 278);
    float3 vup = make_float3(0,1,0);
    float vfov = 40.0f;
    float aspect = float(w)/h;

    float theta = vfov * M_PI / 180.0f;
    float half_h = tan(theta/2);
    float half_w = aspect * half_h;

    float3 cam_w = normalize(lookfrom - lookat);
    float3 cam_u = normalize(cross(vup, cam_w));
    float3 cam_v = cross(cam_w, cam_u);

    float3 llc = lookfrom - cam_u*half_w - cam_v*half_h - cam_w;
    float3 horiz = cam_u * (2*half_w);
    float3 vert = cam_v * (2*half_h);

    float3 pixel_color = make_float3(0,0,0);
    for (int s=0; s<spp; ++s) {
        float u = (col + curand_uniform(&rand_state)) / float(w);
        float v = (row + curand_uniform(&rand_state)) / float(h);
        Ray ray;
        ray.origin = lookfrom;
        ray.direction = normalize(llc + horiz*u + vert*v - lookfrom);
        pixel_color += trace_ray(ray, scene, num_quads, &rand_state);
    }
    pixel_color = pixel_color / spp;
    pixel_color.x = sqrtf(pixel_color.x);
    pixel_color.y = sqrtf(pixel_color.y);
    pixel_color.z = sqrtf(pixel_color.z);
    fb[row*w + col] = pixel_color;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <samples_per_pixel>\n";
        return 1;
    }
    int spp = atoi(argv[1]);
    int w = 512, h = 512;

    vector<Quad> scene(18);
    // Cornell Box
    scene[0] = {0.0f, 0, 0.0f,555.0f,0.0f,555.0f, {1,0,0}, {Material::DIFFUSE, {0.65,0.05,0.05}, {0}}};
    scene[1] = {555.0f,0, 0.0f,555.0f,0.0f,555.0f, {-1,0,0}, {Material::DIFFUSE, {0.12,0.45,0.15}, {0}}};
    scene[2] = {0.0f,1, 0.0f,555.0f,0.0f,555.0f, {0,1,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[3] = {555.0f,1, 0.0f,555.0f,0.0f,555.0f, {0,-1,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[4] = {555.0f,2, 0.0f,555.0f,0.0f,555.0f, {0,0,-1}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[5] = {554.9f,1, 213.0f,343.0f, 227.0f,332.0f, {0,-1,0}, {Material::EMISSIVE, {0}, {15,15,15}}};
    // Box 1
    scene[6] = {130.0f,0, 0.0f,165.0f,65.0f,230.0f, {-1,0,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[7] = {295.0f,0, 0.0f,165.0f,65.0f,230.0f, {1,0,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[8] = {65.0f,2, 130.0f,295.0f,0.0f,165.0f, {0,0,-1}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[9] = {230.0f,2, 130.0f,295.0f,0.0f,165.0f, {0,0,1}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[10] = {0.0f,1, 130.0f,295.0f,65.0f,230.0f, {0,1,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[11] = {165.0f,1, 130.0f,295.0f,65.0f,230.0f, {0,-1,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    // Box 2
    scene[12] = {405.0f,0, 0.0f,330.0f,225.0f,390.0f, {-1,0,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[13] = {555.0f,0, 0.0f,330.0f,225.0f,390.0f, {1,0,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[14] = {225.0f,2, 405.0f,555.0f,0.0f,330.0f, {0,0,-1}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[15] = {390.0f,2, 405.0f,555.0f,0.0f,330.0f, {0,0,1}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[16] = {0.0f,1, 405.0f,555.0f,225.0f,390.0f, {0,1,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};
    scene[17] = {330.0f,1, 405.0f,555.0f,225.0f,390.0f, {0,-1,0}, {Material::DIFFUSE, {0.73,0.73,0.73}, {0}}};

    Quad* d_scene;
    cudaMalloc(&d_scene, scene.size() * sizeof(Quad));
    cudaMemcpy(d_scene, scene.data(), scene.size() * sizeof(Quad), cudaMemcpyHostToDevice);

    float3* fb;
    cudaMallocManaged(&fb, w*h*sizeof(float3));

    dim3 blocks((w+15)/16, (h+15)/16);
    dim3 threads(16,16);
    render_kernel<<<blocks, threads>>>(fb, w, h, spp, d_scene, scene.size());

    
    if (cuErr != cudaSuccess) {
        std::cerr << "CUDA error after kernel: "
                  << cudaGetErrorString(cuErr) << "\n";
        return 1;
    }

    cudaDeviceSynchronize();

    ofstream out("output.ppm");
    out << "P3\n" << w << " " << h << "\n255\n";
    for (int j = h-1; j >=0; --j) {
        for (int i=0; i<w; ++i) {
            size_t idx = j*w + i;
            float r = fminf(fmaxf(fb[idx].x, 0.0f), 1.0f);
            float g = fminf(fmaxf(fb[idx].y, 0.0f), 1.0f);
            float b = fminf(fmaxf(fb[idx].z, 0.0f), 1.0f);
            int ir = int(255.99f * r);
            int ig = int(255.99f * g);
            int ib = int(255.99f * b);
            out << ir << " " << ig << " " << ib << "\n";
        }
    }

    cudaFree(d_scene);
    cudaFree(fb);
    return 0;
}