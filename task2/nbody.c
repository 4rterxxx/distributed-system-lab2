#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define G 6.67430e-11
#define DT 3600.0
#define EPS 1e-10

typedef struct {
    double mass, x, y, z, vx, vy, vz, ax, ay, az;
} Particle;

int main(int argc, char* argv[]) {
    if (argc != 5) {
        printf("Usage: %s <tend> <input_file> <output_file> <threads>\n", argv[0]);
        return 1;
    }
    
    double tend = atof(argv[1]);
    char* input_file = argv[2];
    char* output_file = argv[3];
    int num_threads = atoi(argv[4]);
    
    omp_set_num_threads(num_threads);
    
    // Измерение времени начала выполнения
    double start_time = omp_get_wtime();
    
    // 1. Чтение данных
    FILE* in = fopen(input_file, "r");
    if (!in) {
        printf("Cannot open %s\n", input_file);
        return 1;
    }
    
    int n;
    fscanf(in, "%d", &n);
    
    // Динамическое выделение памяти
    Particle* p = malloc(n * sizeof(Particle));
    
    for (int i = 0; i < n; i++) {
        fscanf(in, "%lf %lf %lf %lf %lf %lf %lf",
               &p[i].mass, &p[i].x, &p[i].y, &p[i].z,
               &p[i].vx, &p[i].vy, &p[i].vz);
        p[i].ax = p[i].ay = p[i].az = 0.0;
    }
    fclose(in);
    
    // 2. Выходной файл
    FILE* out = fopen(output_file, "w");
    if (!out) {
        printf("Cannot create %s\n", output_file);
        free(p);
        return 1;
    }
    
    // Заголовок
    fprintf(out, "t");
    for (int i = 0; i < n; i++) 
        fprintf(out, ",x%d,y%d", i+1, i+1);
    fprintf(out, "\n");
    
    // 3. Начальное состояние
    fprintf(out, "0.0");
    for (int i = 0; i < n; i++) 
        fprintf(out, ",%.6e,%.6e", p[i].x, p[i].y);
    fprintf(out, "\n");
    
    // 4. Симуляция
    int steps = tend / DT;
    
    for (int step = 1; step <= steps; step++) {
        double t = step * DT;
        
        // Ускорения
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            double axi = 0.0, ayi = 0.0, azi = 0.0;
            
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                
                double dx = p[j].x - p[i].x;
                double dy = p[j].y - p[i].y;
                double dz = p[j].z - p[i].z;
                
                double dist2 = dx*dx + dy*dy + dz*dz + EPS;
                double dist = sqrt(dist2);
                double dist3 = dist * dist * dist;
                
                double force = G * p[j].mass / dist3;
                axi += force * dx;
                ayi += force * dy;
                azi += force * dz;
            }
            
            p[i].ax = axi;
            p[i].ay = ayi;
            p[i].az = azi;
        }
        
        // Обновление
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i].vx += p[i].ax * DT;
            p[i].vy += p[i].ay * DT;
            p[i].vz += p[i].az * DT;
            
            p[i].x += p[i].vx * DT;
            p[i].y += p[i].vy * DT;
            p[i].z += p[i].vz * DT;
        }
        
        // Запись
        fprintf(out, "%.1f", t);
        for (int i = 0; i < n; i++) 
            fprintf(out, ",%.6e,%.6e", p[i].x, p[i].y);
        fprintf(out, "\n");
    }
    
    fclose(out);
    free(p);
    
    // Измерение времени окончания и вывод результата
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    
    printf("threads_num %d n_p %d t %.6f\n", num_threads, n, elapsed_time);
    
    return 0;
}