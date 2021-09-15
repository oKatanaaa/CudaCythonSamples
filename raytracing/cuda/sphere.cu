#define rnd( x ) (x * rand() / RAND_MAX)
#define INF 1e10f

struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;

    /**
    * Checks whether a ray shot from the pixel at (ox, oy)
    * hits the sphere.
    *
    * @param ox The x coordinate of the pixel the ray were shot from.
    * @param oy The y coordinate of the pixel the ray were shot from.
    * @param intensity The intensity of the color.
    * @return Distance from the point on the sphere hit by the ray
    * to the camera.
    */
    __device__ float hit(float ox, float oy, float *intensity) {
        float dx = ox - x;
        float dy = oy - y;
        if (dx*dx + dy*dy < radius * radius) {
            float dz = sqrtf(radius*radius - dx*dx - dy*dy);
            // The closer the hit point to the circle border,
            // the darker it is.
            *intensity = dz / radius;
            return dz + z;
        }
        return -INF;
    }
};


Sphere* generate_random_spheres(int n_spheres, int w, int h) {
    Sphere* spheres = new Sphere[n_spheres];
    float f_w = (float) w, f_h = (float) h;
    for (int i = 0; i < n_spheres; i++) {
        spheres[i].r = rnd(1.0f);
        spheres[i].g = rnd(1.0f);
        spheres[i].b = rnd(1.0f);

        spheres[i].x = rnd(f_w) - f_w / 2;
        spheres[i].y = rnd(f_h) - f_h / 2;
        spheres[i].z = rnd(1000.0f) - 500;
        spheres[i].radius = rnd(100.0f) + 20;
    }
    return spheres;
}
