set title "Execution Time" 
set ylabel "Time (Seconds)"
set xlabel "Size"

set style line 1 lt 2 lc rgb "cyan"   lw 2 
set style line 2 lt 2 lc rgb "red"    lw 2
set style line 3 lt 2 lc rgb "gold"   lw 2
set style line 4 lt 2 lc rgb "green"  lw 2
set style line 5 lt 2 lc rgb "blue"   lw 2
set style line 6 lt 2 lc rgb "black"  lw 2
set terminal postscript eps enhanced color
set output 'time.eps'

set xtics nomirror
set ytics nomirror
set key top left
set key box
set style data lines


plot "file_comparison.data" using 1:2 title "Sequential"                  ls 1 with linespoints,\
     "file_comparison.data" using 1:3 title "CUDA"                        ls 2 with linespoints,\
     "file_comparison.data" using 1:4 title "OMP5"                        ls 3 with linespoints,\
     "file_comparison.data" using 1:5 title "ACC"                         ls 4 with linespoints