gcc -fopenmp -O3 -o nbody nbody.c -lm

for file in 50_p.txt 500_p.txt 5000_p.txt
do
    if [ -f "$file" ]
    then
        particles=$(echo "$file" | grep -oE '[0-9]+')
        for threads in 1 2 4 8
        do
            ./nbody 36000.0 "$file" "output_${particles}_${threads}.csv" "$threads"
        done
    fi
done