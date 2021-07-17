#!/bin/sh

###################################
# START SCRIPT                    #
###################################

clear 

mostraHelice() {
   
   tput civis
   
   while [ -d /proc/$! ]
   do
      for i in / - \\ \|
      do
         printf "\033[1D$i"
         sleep .1
      done
   done
   
   tput cnorm
}

usage()
{
 echo "START.sh: wrong number of input parameters. Exiting."
 echo -e "Usage: bash START.sh <supercomputer> [[[--comparison file ] | [--help]]"
 echo -e "  g.e: bash START.sh ogbon"
}


airis()
{
 module load gcc/8.3.0 
 module load llvm/10.0.0
 module load pgi/2019 
 module load gnuplot/5.2.8
 module load blas/gcc/3.8.0
 export PATH=/opt/share/llvm_bkp/bin:$PATH && export LIBRARY_PATH=/opt/share/llvm_bkp/lib:/opt/share/gcc/8.3.0/lib:/opt/share/gcc/8.3.0/lib64:$LIBRARY_PATH && export LD_LIBRARY_PATH=/opt/share/llvm_bkp/lib:/opt/share/gcc/8.3.0/lib:/opt/share/gcc/8.3.0/lib64:$LD_LIBRARY_PATH
 gcc mm.c -o mm -fopenmp 
 gcc mm_blas.c -o mm_blas -lblas -lgfortran -fopenmp
 nvcc mm_cuda.cu -o mm_cuda -Xcompiler -fopenmp 
 nvcc mm_cublas.cu -o mm_cublas -Xcompiler -fopenmp -lcublas
 clang mm_omp5.c -o mm_omp5 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
 pgcc mm_openacc.c -o mm_openacc -acc
}

ogbon()
{
 module load gnuplot/5.2.8 
 module load gcc/8.3.1 
 gcc mm.c -o mm -fopenmp 
 sleep .1
 module load blas/gcc/3.8.0
 gcc mm_blas.c -o mm_blas -lblas -lgfortran -fopenmp
 sleep .1
 module load cuda/10.1  
 nvcc mm_cuda.cu -o mm_cuda -Xcompiler -fopenmp 
 nvcc mm_cublas.cu -o mm_cublas -Xcompiler -fopenmp -lcublas
 sleep .1
 module load llvm/10.0.0 
 clang mm_omp5.c -o mm_omp5 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda
 sleep .1
 module load pgi/19.10 
 pgcc mm_openacc.c -o mm_openacc -acc
 sleep .1
 module unload pgi/19.10 
}

label_experimental_time()
{

 local filename=$1    
  
 echo " "
 echo " "
 echo "    ********************************"
 echo "    * Experimental Time Comparison *"
 echo "    ********************************"
 echo " "

 if [ "$filename" == "mm" ]; then
  echo "    [#][size]       [S]            [CUDA]          [OMP5]          [ACC]"
 fi

 if [ "$filename" == "mm_blas" ]; then
  echo "    [#][size]      [BLAS]         [CUDA]          [OMP5]          [ACC]"
 fi

 if [ "$filename" == "mm_cublas" ]; then
  echo "    [#][size]      [CUBLAS]         [CUDA]          [OMP5]          [ACC]"
 fi


}

label_speedup()
{
 echo " "
 echo " "
 echo "    ********************************"
 echo "    * Speedup  Rate                *"
 echo "    ********************************"
 echo " "
 echo "    [#][size]    [SCUDA]         [SOMP5]         [SACC]"

}

create_scripts_gnuplot_time()
{
local variable=$1 

if [[ $variable == "mm" ]];then
      sequential=Sequential
fi

if [[ $variable == "mm_blas" ]];then
      sequential=BLAS
fi

if [[ $variable == "mm_cublas" ]];then
      sequential=CUBLAS
fi

cat << EOF > time.plt 
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

plot "file_comparison.data" using 1:2 title "$sequential"                 ls 1 with linespoints,\
     "file_comparison.data" using 1:3 title "CUDA"                        ls 2 with linespoints,\
     "file_comparison.data" using 1:4 title "OMP5"                        ls 3 with linespoints,\
     "file_comparison.data" using 1:5 title "ACC"                         ls 4 with linespoints
EOF

}

create_scripts_gnuplot_speedup()
{
cat << EOF > speedup.plt 
set title "Speedup" 
set ylabel "Speedup"
set xlabel "Size"

set style line 1 lt 2 lc rgb "cyan"   lw 2 
set style line 2 lt 2 lc rgb "red"    lw 2
set style line 3 lt 2 lc rgb "gold"   lw 2
set style line 4 lt 2 lc rgb "green"  lw 2
set style line 5 lt 2 lc rgb "blue"   lw 2
set style line 6 lt 2 lc rgb "black"  lw 2
set terminal postscript eps enhanced color
set output 'speedup.eps'

set xtics nomirror
set ytics nomirror
set key top left
set key box
set style data lines

plot "file_speedup.data" using 1:2 title "CUDA"        ls 2 with linespoints,\
     "file_speedup.data" using 1:3 title "OMP5"        ls 3 with linespoints,\
     "file_speedup.data" using 1:4 title "ACC"         ls 4 with linespoints
EOF

}

create_scripts_gnuplot_memory()
{
cat << EOF > memory.plt 
set title "Memory" 
set xlabel "Timeline"
set ylabel "Memory (Kb)"

set xtics nomirror
set ytics nomirror
set key top left
set key box
set style data lines

set style line 1 lt 2 lc rgb "cyan"   lw 2 
set style line 2 lt 2 lc rgb "red"    lw 2
set style line 3 lt 2 lc rgb "gold"   lw 2
set style line 4 lt 2 lc rgb "green"  lw 2
set style line 5 lt 2 lc rgb "blue"   lw 2
set style line 6 lt 2 lc rgb "black"  lw 2

set terminal postscript eps enhanced color
set output 'memory.eps'

plot  "file_memory.data"     using 1:2 title "CUDA"    ls 2 with linespoints, \
       "file_memory.data"    using 1:3 title "OMP5"    ls 3 with linespoints, \
       "file_memory.data"    using 1:4 title "ACC"     ls 4 with linespoints
EOF

}

sleep 3 > /dev/null 2>&1 &

printf "Loading...\040\040" ; mostraHelice
echo " "

#######################################################
# COMPILATION + PERMISSIONS  TO EXECUTE + ENVIRONMENT #
#######################################################

#args in comand line
if [ "$#" ==  2 ]; then
 usage
 exit
fi

#airis
if [[ $1 == "airis" ]];then
 airis
fi


#ogbon
if [[ $1 == "ogbon" ]];then
 ogbon
fi

chmod +x mm
chmod +x mm_cuda
chmod +x mm_omp5
chmod +x mm_openacc
chmod +x mm_blas
chmod +x mm_cublas

###################################
# EXPERIMENTAL TIMES              #
###################################

filename=mm


while [ "$2" != "" ]; do
    case $2 in
        -c | --comparison )           shift
                                filename=$2
                                ;;
        -h | --help )           usage
                                exit                        
    esac
    shift
done
echo " "
echo "sequential = [$filename]"


for model in 'sequential' 'cuda' 'omp5' 'openacc'
do
  nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits --loop-ms=20 --filename=outputMemGPU-smi-$model.txt > /dev/null 2>&1 &
  PID=$!
  echo ""
  echo "$model:"
  for i in 100 200 300 400 500 600 700 800 900 1000
  do
    printf "\033[1D$i :" 
    if [[ $model =  'sequential' ]]; then
      ./$filename $i >> file
    else
      ./mm_$model $i >> file-$model
    fi
  done
  sleep 1
  kill $PID
  sleep 1
  awk '{print $1}' outputMemGPU-smi-$model.txt > aux-$model.txt
  awk '{print FNR " " $1}' aux-$model.txt > outputUseMemory-$model
done

pr -m -t -s\  outputUseMemory-cuda outputUseMemory-omp5 outputUseMemory-openacc  | awk '{print $1,"\t",$2,"\t",$4,"\t",$6}' > file_memory.data

clear 

#####################
# TIME              #
#####################

label_experimental_time $filename

pr -m -t -s\  file file-cuda file-omp5 file-openacc  | awk '{print $1,"\t",$2,"\t",$4,"\t",$6,"\t",$8}' > file_comparison.data

cat -n  file_comparison.data

sleep 1

#####################
# SPEEDUP           #
#####################

awk '{print $1, " ",(($2*1000)/($3*1000))}' file_comparison.data > fspeed0 
awk '{print $1, " ",(($2*1000)/($4*1000))}' file_comparison.data > fspeed1 
awk '{print $1, " ",(($2*1000)/($5*1000))}' file_comparison.data > fspeed2 


pr -m -t -s\  fspeed0 fspeed1 fspeed2 | awk '{print $1,"\t",$2,"\t",$4,"\t",$6}' > file_speedup.data

label_speedup

cat -n file_speedup.data

sleep 1

#####################
# PLOTING           #
#####################

echo " "
echo "Do you want to plot a graphic (y/n)?"
read resp1

if [[ $resp1 == "y" ]];then
     
#create plots         
  create_scripts_gnuplot_time $filename
  create_scripts_gnuplot_speedup
  create_scripts_gnuplot_memory
         
  sleep .1

  echo "ploting eps graphic with gnuplot..."
  gnuplot "time.plt"
  gnuplot "speedup.plt"
  gnuplot "memory.plt"

#rename output
  mv time.eps     time.$(whoami)@$(hostname)-$(date +%F%T).eps
  mv speedup.eps  speedup.$(whoami)@$(hostname)-$(date +%F%T).eps
  mv memory.eps   memory.$(whoami)@$(hostname)-$(date +%F%T).eps
fi

sleep 1

###################################
# NVPROOF TESTS                   #
###################################

echo " "
echo "Do you want to execute profiling (y/n)?"
read resp2

if [[ $resp2 == "y" ]];then
  n=1000
  echo "profiling with NVPROF..."
  for model in 'cuda' 'omp5' 'openacc'
  do 
    sleep 1
    nvprof --log-file profile-gpu-$model-$(whoami)@$(hostname)-$(date +%F%T).log ./mm_$model $n > /dev/null 2>&1 &     
    sleep 1
  done
sleep 1
fi


###################################
# DELETE FILES                    #
###################################

echo " "
echo "[Remove unnecessary files] "
rm -f *.txt file* fspeed* *.data mm mm_blas mm_cuda mm_omp5 mm_openacc outputUseMemory-* *.plt mm_cublas
mkdir  -p results-$(whoami)@$(hostname)-$(date +%F)
if [[ $resp1 == "y" ]];then
 mv *.eps  results-$(whoami)@$(hostname)-$(date +%F)
fi

if [[ $resp2 == "y" ]];then
 mv *.log results-$(whoami)@$(hostname)-$(date +%F)
fi


###################################
# END SCRIPT                      #
###################################

echo " "

sleep 5 > /dev/null 2>&1 &

printf "Loading...\040\040" ; mostraHelice
echo " "
echo "[END] " 
echo " "



