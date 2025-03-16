#include <math.h>

__device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(float3 v)
{
    float len = sqrtf(dot(v, v));
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ float3 reflect(float3 v, float3 n)
{
    return make_float3(v.x - 2.0f * dot(v, n) * n.x, v.y - 2.0f * dot(v, n) * n.y, v.z - 2.0f * dot(v, n) * n.z);
}

__device__ float3 mix(float3 a, float3 b, float t)
{
    return make_float3(a.x * (1.0f - t) + b.x * t, a.y * (1.0f - t) + b.y * t, a.z * (1.0f - t) + b.z * t);
}

__device__ bool intersect_sphere(float3 ro, float3 rd, float3 sphere_c, float sphere_r, float &t_hit)
{
    float3 oc = make_float3(ro.x - sphere_c.x, ro.y - sphere_c.y, ro.z - sphere_c.z);
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sphere_r * sphere_r;
    
    if (b * b - c < 0.0f) return false;
    
    float disc = sqrtf(b * b - c);
    float t0 = -b - disc;
    float t1 = -b + disc;
    
    if (t0 > 0.0f && t1 > 0.0f) 
    {
        t_hit = fminf(t0, t1);
        return true;
    } 
    else if (t0 > 0.0f) 
    {
        t_hit = t0;
        return true;
    } 
    else if (t1 > 0.0f) 
    {
        t_hit = t1;
        return true;
    }
    
    return false;
}

__device__ bool intersect_plane(float3 ro, float3 rd, float3 p_normal, float p_dist, float &t_hit)
{
    float denom = dot(p_normal, rd);
    
    if (fabsf(denom) < 1e-6f) return false;
    
    t_hit = -(dot(p_normal, ro) + p_dist) / denom;
    return t_hit > 0.0f;
}

__device__ float3 floor_pattern(float x, float z)
{
    int check = (int(floorf(x)) + int(floorf(z))) % 2;
    return check ? make_float3(0.8f, 0.8f, 0.8f) : make_float3(0.3f, 0.3f, 0.3f);
}

__device__ float3 sky_color(float3 dir)
{
    float t = 0.5f * (dir.y + 1.0f);
    float3 sky_blue = make_float3(0.5f, 0.7f, 1.0f);
    float3 white = make_float3(1.0f, 1.0f, 1.0f);
    return mix(white, sky_blue, t);
}

__device__ void collision(float3 &sphere1_pos, float3 &sphere2_pos, float3 &sphere3_pos, float3 &sphere1_vel, float3 &sphere2_vel, float3 &sphere3_vel, float sphere1_radius, float sphere2_radius, float sphere3_radius, float time_step, float time)
{
    float gravity = 0.5f;
    sphere1_vel.y -= gravity * time_step;
    sphere2_vel.y -= gravity * time_step;
    sphere3_vel.y -= gravity * time_step;
    
    if (sphere1_pos.y - sphere1_radius < 0.0f) 
    {
        sphere1_pos.y = sphere1_radius;
        sphere1_vel.y = -sphere1_vel.y * 0.8f;
        sphere1_vel.x *= 0.95f;
        sphere1_vel.z *= 0.95f;
    }
    
    if (sphere2_pos.y - sphere2_radius < 0.0f) 
    {
        sphere2_pos.y = sphere2_radius;
        sphere2_vel.y = -sphere2_vel.y * 0.8f;
        sphere2_vel.x *= 0.95f;
        sphere2_vel.z *= 0.95f;
    }
    
    if (sphere3_pos.y - sphere3_radius < 0.0f) 
    {
        sphere3_pos.y = sphere3_radius;
        sphere3_vel.y = -sphere3_vel.y * 0.8f;
        sphere3_vel.x *= 0.95f;
        sphere3_vel.z *= 0.95f;
    }
    
    float3 diff12 = make_float3(sphere1_pos.x - sphere2_pos.x, sphere1_pos.y - sphere2_pos.y, sphere1_pos.z - sphere2_pos.z);
    float dist12_sq = dot(diff12, diff12);
    float min_dist12 = sphere1_radius + sphere2_radius;
    
    if (dist12_sq < min_dist12 * min_dist12) 
    {
        float dist12 = sqrtf(dist12_sq);
        float3 normal = make_float3(diff12.x / dist12, diff12.y / dist12, diff12.z / dist12);
        
        float overlap = min_dist12 - dist12;
        float mass1 = sphere1_radius * sphere1_radius * sphere1_radius;
        float mass2 = sphere2_radius * sphere2_radius * sphere2_radius;
        float total_mass = mass1 + mass2;
        
        sphere1_pos.x += normal.x * overlap * (mass2 / total_mass);
        sphere1_pos.y += normal.y * overlap * (mass2 / total_mass);
        sphere1_pos.z += normal.z * overlap * (mass2 / total_mass);
        
        sphere2_pos.x -= normal.x * overlap * (mass1 / total_mass);
        sphere2_pos.y -= normal.y * overlap * (mass1 / total_mass);
        sphere2_pos.z -= normal.z * overlap * (mass1 / total_mass);
        
        float v_rel_x = sphere1_vel.x - sphere2_vel.x;
        float v_rel_y = sphere1_vel.y - sphere2_vel.y;
        float v_rel_z = sphere1_vel.z - sphere2_vel.z;
        
        float v_rel_dot_n = v_rel_x * normal.x + v_rel_y * normal.y + v_rel_z * normal.z;
        
        if (v_rel_dot_n < 0.0f) 
        {
            float j = -(1.8f) * v_rel_dot_n;
            j /= (1.0f / mass1) + (1.0f / mass2);
            
            float impulse_x = j * normal.x;
            float impulse_y = j * normal.y;
            float impulse_z = j * normal.z;
            
            sphere1_vel.x += impulse_x / mass1;
            sphere1_vel.y += impulse_y / mass1;
            sphere1_vel.z += impulse_z / mass1;
            
            sphere2_vel.x -= impulse_x / mass2;
            sphere2_vel.y -= impulse_y / mass2;
            sphere2_vel.z -= impulse_z / mass2;
        }
    }
    
    float3 diff13 = make_float3(sphere1_pos.x - sphere3_pos.x, sphere1_pos.y - sphere3_pos.y, sphere1_pos.z - sphere3_pos.z);
    float dist13_sq = dot(diff13, diff13);
    float min_dist13 = sphere1_radius + sphere3_radius;
    
    if (dist13_sq < min_dist13 * min_dist13) 
    {
        float dist13 = sqrtf(dist13_sq);
        float3 normal = make_float3(diff13.x / dist13, diff13.y / dist13, diff13.z / dist13);
        
        float overlap = min_dist13 - dist13;
        float mass1 = sphere1_radius * sphere1_radius * sphere1_radius;
        float mass3 = sphere3_radius * sphere3_radius * sphere3_radius;
        float total_mass = mass1 + mass3;
        
        sphere1_pos.x += normal.x * overlap * (mass3 / total_mass);
        sphere1_pos.y += normal.y * overlap * (mass3 / total_mass);
        sphere1_pos.z += normal.z * overlap * (mass3 / total_mass);
        
        sphere3_pos.x -= normal.x * overlap * (mass1 / total_mass);
        sphere3_pos.y -= normal.y * overlap * (mass1 / total_mass);
        sphere3_pos.z -= normal.z * overlap * (mass1 / total_mass);
        
        float v_rel_x = sphere1_vel.x - sphere3_vel.x;
        float v_rel_y = sphere1_vel.y - sphere3_vel.y;
        float v_rel_z = sphere1_vel.z - sphere3_vel.z;
        
        float v_rel_dot_n = v_rel_x * normal.x + v_rel_y * normal.y + v_rel_z * normal.z;
        
        if (v_rel_dot_n < 0.0f) 
        {
            float j = -(1.8f) * v_rel_dot_n;
            j /= (1.0f / mass1) + (1.0f / mass3);
            
            float impulse_x = j * normal.x;
            float impulse_y = j * normal.y;
            float impulse_z = j * normal.z;
            
            sphere1_vel.x += impulse_x / mass1;
            sphere1_vel.y += impulse_y / mass1;
            sphere1_vel.z += impulse_z / mass1;
            
            sphere3_vel.x -= impulse_x / mass3;
            sphere3_vel.y -= impulse_y / mass3;
            sphere3_vel.z -= impulse_z / mass3;
        }
    }
    
    float3 diff23 = make_float3(sphere2_pos.x - sphere3_pos.x, sphere2_pos.y - sphere3_pos.y, sphere2_pos.z - sphere3_pos.z);
    float dist23_sq = dot(diff23, diff23);
    float min_dist23 = sphere2_radius + sphere3_radius;
    
    if (dist23_sq < min_dist23 * min_dist23) 
    {
        float dist23 = sqrtf(dist23_sq);
        float3 normal = make_float3(diff23.x / dist23, diff23.y / dist23, diff23.z / dist23);
        
        float overlap = min_dist23 - dist23;
        float mass2 = sphere2_radius * sphere2_radius * sphere2_radius;
        float mass3 = sphere3_radius * sphere3_radius * sphere3_radius;
        float total_mass = mass2 + mass3;
        
        sphere2_pos.x += normal.x * overlap * (mass3 / total_mass);
        sphere2_pos.y += normal.y * overlap * (mass3 / total_mass);
        sphere2_pos.z += normal.z * overlap * (mass3 / total_mass);
        
        sphere3_pos.x -= normal.x * overlap * (mass2 / total_mass);
        sphere3_pos.y -= normal.y * overlap * (mass2 / total_mass);
        sphere3_pos.z -= normal.z * overlap * (mass2 / total_mass);

        float v_rel_x = sphere2_vel.x - sphere3_vel.x;
        float v_rel_y = sphere2_vel.y - sphere3_vel.y;
        float v_rel_z = sphere2_vel.z - sphere3_vel.z;
        
        float v_rel_dot_n = v_rel_x * normal.x + v_rel_y * normal.y + v_rel_z * normal.z;
        
        if (v_rel_dot_n < 0.0f) 
        {
            float j = -(1.8f) * v_rel_dot_n;
            j /= (1.0f / mass2) + (1.0f / mass3);
            
            float impulse_x = j * normal.x;
            float impulse_y = j * normal.y;
            float impulse_z = j * normal.z;
            
            sphere2_vel.x += impulse_x / mass2;
            sphere2_vel.y += impulse_y / mass2;
            sphere2_vel.z += impulse_z / mass2;
            
            sphere3_vel.x -= impulse_x / mass3;
            sphere3_vel.y -= impulse_y / mass3;
            sphere3_vel.z -= impulse_z / mass3;
        }
    }
    
    float boundary = 50.0f;
    
    if (fabsf(sphere1_pos.x) > boundary - sphere1_radius) 
    {
        sphere1_pos.x = (sphere1_pos.x > 0) ? (boundary - sphere1_radius) : -(boundary - sphere1_radius);
        sphere1_vel.x *= -0.8f;
    }
    if (fabsf(sphere1_pos.z) > boundary - sphere1_radius) 
    {
        sphere1_pos.z = (sphere1_pos.z > 0) ? (boundary - sphere1_radius) : -(boundary - sphere1_radius);
        sphere1_vel.z *= -0.8f;
    }
    
    if (fabsf(sphere2_pos.x) > boundary - sphere2_radius) 
    {
        sphere2_pos.x = (sphere2_pos.x > 0) ? (boundary - sphere2_radius) : -(boundary - sphere2_radius);
        sphere2_pos.x = (sphere2_pos.x > 0) ? (boundary - sphere2_radius) : -(boundary - sphere2_radius);
        sphere2_vel.x *= -0.8f;
    }
    if (fabsf(sphere2_pos.z) > boundary - sphere2_radius) 
    {
        sphere2_pos.z = (sphere2_pos.z > 0) ? (boundary - sphere2_radius) : -(boundary - sphere2_radius);
        sphere2_vel.z *= -0.8f;
    }
    
    if (fabsf(sphere3_pos.x) > boundary - sphere3_radius) 
    {
        sphere3_pos.x = (sphere3_pos.x > 0) ? (boundary - sphere3_radius) : -(boundary - sphere3_radius);
        sphere3_vel.x *= -0.8f;
    }
    if (fabsf(sphere3_pos.z) > boundary - sphere3_radius) 
    {
        sphere3_pos.z = (sphere3_pos.z > 0) ? (boundary - sphere3_radius) : -(boundary - sphere3_radius);
        sphere3_vel.z *= -0.8f;
    }

    sphere1_pos.x += sphere1_vel.x * time_step;
    sphere1_pos.y += sphere1_vel.y * time_step;
    sphere1_pos.z += sphere1_vel.z * time_step;
    
    sphere2_pos.x += sphere2_vel.x * time_step;
    sphere2_pos.y += sphere2_vel.y * time_step;
    sphere2_pos.z += sphere2_vel.z * time_step;
    
    sphere3_pos.x += sphere3_vel.x * time_step;
    sphere3_pos.y += sphere3_vel.y * time_step;
    sphere3_pos.z += sphere3_vel.z * time_step;
}

__global__ void render(float *disp, int img_w, int img_h, float time)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= img_h || col >= img_w)
    {
        return;
    }

    int pix_location = row * img_w * 3 + col * 3;
    
    float aspect_ratio = float(img_w) / float(img_h);

    float u = (2.0f * (col + 0.5f) / float(img_w) - 1.0f) * aspect_ratio;
    float v = 1.0f - 2.0f * (row + 0.5f) / float(img_h);

    float3 cam_pos = make_float3(0.0f, 2.0f, -8.0f);
    float3 look_at = make_float3(0.0f, 0.0f, 0.0f);
    float3 up = make_float3(0.0f, 1.0f, 0.0f);
    
    float3 cam_dir = normalize(make_float3(look_at.x - cam_pos.x, look_at.y - cam_pos.y, look_at.z - cam_pos.z));
    
    float3 cam_right = normalize(make_float3(cam_dir.z, 0.0f, -cam_dir.x));
    
    float3 cam_up = normalize(make_float3(cam_dir.y * cam_right.z - cam_dir.z * cam_right.y, cam_dir.z * cam_right.x - cam_dir.x * cam_right.z, cam_dir.x * cam_right.y - cam_dir.y * cam_right.x));
    
    float focal_length = 1.0f;
    float3 ray_dir = normalize(make_float3(cam_dir.x * focal_length + cam_right.x * u + cam_up.x * v, cam_dir.y * focal_length + cam_right.y * u + cam_up.y * v, cam_dir.z * focal_length + cam_right.z * u + cam_up.z * v));

    float3 light_pos = make_float3(5.0f, 10.0f, -5.0f);
    float3 surrounding_light = make_float3(0.1f, 0.1f, 0.1f);
    
    float3 sphere1_pos = make_float3(2.0f, 3.0f, 0.0f);
    float sphere1_radius = 1.0f;
    float3 sphere1_color = make_float3(0.9f, 0.2f, 0.2f);
    float3 sphere1_vel = make_float3(-1.0f, 0.0f, 0.5f);
    
    float3 sphere2_pos = make_float3(-1.5f, 2.0f, 1.0f);
    float sphere2_radius = 0.7f;
    float3 sphere2_color = make_float3(0.2f, 0.2f, 0.9f);
    float3 sphere2_vel = make_float3(0.8f, 0.0f, -0.3f);
    
    float3 sphere3_pos = make_float3(0.0f, 1.0f, 3.0f);
    float sphere3_radius = 0.4f;
    float3 sphere3_color = make_float3(0.3f, 0.8f, 0.3f);
    float3 sphere3_vel = make_float3(0.2f, 0.0f, -0.9f); 
    
    float total_time = time;
    float time_step = 0.01f;
    int steps = int(total_time / time_step);
    
    sphere1_vel.x += (u * 0.01f);
    sphere1_vel.z += (v * 0.01f);
    
    for (int i = 0; i < steps; i++) 
    {
        collision(sphere1_pos, sphere2_pos, sphere3_pos, sphere1_vel, sphere2_vel, sphere3_vel, sphere1_radius, sphere2_radius, sphere3_radius, time_step, float(i) * time_step);
    }

    float3 floor_normal = make_float3(0.0f, 1.0f, 0.0f);
    float floor_dist = 0.0f;
    
    float t_closest = 1e10f;
    float3 color = sky_color(ray_dir);
    float3 normal;
    float3 hit_point;
    float3 material_color;
    float specular_power = 0.0f;
    
    float t_hit;
    bool hit_something = false;
    
    if (intersect_sphere(cam_pos, ray_dir, sphere1_pos, sphere1_radius, t_hit) && t_hit < t_closest) 
    {
        t_closest = t_hit;
        hit_point = make_float3(cam_pos.x + t_hit * ray_dir.x, cam_pos.y + t_hit * ray_dir.y, cam_pos.z + t_hit * ray_dir.z);
        normal = normalize(make_float3(hit_point.x - sphere1_pos.x, hit_point.y - sphere1_pos.y, hit_point.z - sphere1_pos.z));
        material_color = sphere1_color;
        specular_power = 32.0f;
        hit_something = true;
    }
    
    if (intersect_sphere(cam_pos, ray_dir, sphere2_pos, sphere2_radius, t_hit) && t_hit < t_closest) 
    {
        t_closest = t_hit;
        hit_point = make_float3(cam_pos.x + t_hit * ray_dir.x, cam_pos.y + t_hit * ray_dir.y, cam_pos.z + t_hit * ray_dir.z);
        normal = normalize(make_float3(hit_point.x - sphere2_pos.x, hit_point.y - sphere2_pos.y, hit_point.z - sphere2_pos.z));
        material_color = sphere2_color;
        specular_power = 64.0f;
        hit_something = true;
    }
    
    if (intersect_sphere(cam_pos, ray_dir, sphere3_pos, sphere3_radius, t_hit) && t_hit < t_closest) 
    {
        t_closest = t_hit;
        hit_point = make_float3(cam_pos.x + t_hit * ray_dir.x, cam_pos.y + t_hit * ray_dir.y, cam_pos.z + t_hit * ray_dir.z);
        normal = normalize(make_float3(hit_point.x - sphere3_pos.x, hit_point.y - sphere3_pos.y, hit_point.z - sphere3_pos.z));
        material_color = sphere3_color;
        specular_power = 16.0f;
        hit_something = true;
    }
    
    if (intersect_plane(cam_pos, ray_dir, floor_normal, floor_dist, t_hit) && t_hit < t_closest) 
    {
        t_closest = t_hit;
        hit_point = make_float3(cam_pos.x + t_hit * ray_dir.x, cam_pos.y + t_hit * ray_dir.y, cam_pos.z + t_hit * ray_dir.z);
        normal = floor_normal;
        material_color = floor_pattern(hit_point.x, hit_point.z);
        specular_power = 4.0f;
        hit_something = true;
    }
    
    if (hit_something) 
    {
        float3 light_dir = normalize(make_float3(light_pos.x - hit_point.x, light_pos.y - hit_point.y, light_pos.z - hit_point.z));
        
        bool in_shadow = false;
        float t_shadow;
        
        if (intersect_sphere(hit_point, light_dir, sphere1_pos, sphere1_radius, t_shadow) || intersect_sphere(hit_point, light_dir, sphere2_pos, sphere2_radius, t_shadow) || intersect_sphere(hit_point, light_dir, sphere3_pos, sphere3_radius, t_shadow)) 
        {
            in_shadow = true;
        }

        float diffuse = fmaxf(0.0f, dot(normal, light_dir));

        float3 reflect_dir = reflect(make_float3(-light_dir.x, -light_dir.y, -light_dir.z), normal);
        float spec = powf(fmaxf(0.0f, dot(reflect_dir, make_float3(-ray_dir.x, -ray_dir.y, -ray_dir.z))), specular_power);
        
        if (in_shadow) 
        {
            color = make_float3(material_color.x * surrounding_light.x, material_color.y * surrounding_light.y, material_color.z * surrounding_light.z);
        } 
        else 
        {
            color = make_float3(fminf(1.0f, material_color.x * (surrounding_light.x + diffuse) + spec), fminf(1.0f, material_color.y * (surrounding_light.y + diffuse) + spec), fminf(1.0f, material_color.z * (surrounding_light.z + diffuse) + spec));
        }
    }
    
    color.x = powf(color.x, 1.0f/2.2f);
    color.y = powf(color.y, 1.0f/2.2f);
    color.z = powf(color.z, 1.0f/2.2f);
    
    disp[pix_location + 0] = color.x;
    disp[pix_location + 1] = color.y;
    disp[pix_location + 2] = color.z;
}