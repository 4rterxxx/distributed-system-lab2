#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_ITER 1000

typedef struct {
    double x;
    double y;
} Point;

bool is_mandelbrot_point(double complex c) {
    double complex z = 0.0 + 0.0 * I;
    for (int i = 0; i < MAX_ITER; i++) {
        z = z * z + c;
        if (cabs(z) >= 2.0) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    // проверка аргументов командной строки
    if (argc != 3) {
        fprintf(stderr, "использование %s <nthreads> <npoints>\n", argv[0]);
        fprintf(stderr, "  nthreads - количество потоков\n");
        fprintf(stderr, "  npoints  - количество точек\n");
        return 1;
    }
    
    int nthreads = atoi(argv[1]);
    int npoints = atoi(argv[2]);
    
    if (nthreads <= 0 || npoints <= 0) {
        fprintf(stderr, "error: nthreads и npoints должны быть положительными\n");
        return 1;
    }
    
    // установка количества потоков OpenMP
    omp_set_num_threads(nthreads);
    
    // область визуализации множества Мандельброта
    const double x_min = -2.0;
    const double x_max = 1.0;
    const double y_min = -1.5;
    const double y_max = 1.5;
    
    // создаем равномерную сетку
    int grid_side = (int)sqrt((double)npoints);
    if (grid_side < 1) grid_side = 1;
    
    int actual_points = grid_side * grid_side;
    if (actual_points != npoints) {
        printf("количество точек изменено с %d на %d\n", 
               npoints, actual_points);
        printf("квадрат %dx%d\n", grid_side, grid_side);
        npoints = actual_points;
    }
    
    Point *results = (Point*)malloc(npoints * sizeof(Point));
    if (results == NULL) {
        fprintf(stderr, "ошибка выделения памяти\n");
        return 1;
    }
    
    int total_found = 0;
    
    // начало измерения времени
    double start_time = omp_get_wtime();
    
    #pragma omp parallel
    {
        // локальный массив для каждого потока
        Point *local_points = (Point*)malloc(npoints * sizeof(Point));
        int local_count = 0;
        
        // размер шага
        double x_step = (x_max - x_min) / (grid_side - 1);
        double y_step = (y_max - y_min) / (grid_side - 1);
        
        // динамическое распределение строк сетки по потокам
        #pragma omp for schedule(dynamic, 10)
        for (int i = 0; i < grid_side; i++) {
            double y = y_min + i * y_step;
            
            for (int j = 0; j < grid_side; j++) {
                double x = x_min + j * x_step;
                double complex c = x + y * I;
                
                if (is_mandelbrot_point(c)) {
                    local_points[local_count].x = x;
                    local_points[local_count].y = y;
                    local_count++;
                }
            }
        }
        
        // объединение результатов из всех потоков
        #pragma omp critical
        {
            for (int k = 0; k < local_count; k++) {
                results[total_found + k] = local_points[k];
            }
            total_found += local_count;
        }
        
        free(local_points);
    }
    
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;
    
    char filename[100];
    snprintf(filename, sizeof(filename), 
             "mandelbrot_%d_%d.csv", nthreads, npoints);
    
    FILE *csv_file = fopen(filename, "w");
    if (csv_file == NULL) {
        fprintf(stderr, "ошибка создания файла %s\n", filename);
        free(results);
        return 1;
    }
    
    fprintf(csv_file, "x,y\n");
    
    for (int i = 0; i < total_found; i++) {
        fprintf(csv_file, "%.15f,%.15f\n", results[i].x, results[i].y);
    }
    
    fclose(csv_file);
    
    printf("Параллельное вычисление множества Мандельброта\n");
    printf("количество потоков: %d\n", nthreads);
    printf("размер сетки: %dx%d\n", grid_side, grid_side);
    printf("всего проверено точек: %d\n", npoints);
    printf("точек в множестве: %d\n", total_found);
    printf("процент точек в мн-ве: %.2f%%\n", 
           (double)total_found / npoints * 100.0);
    printf("время выполнения: %.4f секунд\n", execution_time);
    printf("среднее время на точку: %.6f мс\n", 
           execution_time / npoints * 1000.0);
    printf("файл с результатами: %s\n", filename);
    
    free(results);
    return 0;
}