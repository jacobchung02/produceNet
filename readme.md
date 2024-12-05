
## Project Description

<h4>This project constructs an image classification system with both CPU and GPU-driven multi-threaded processing capabilities to sort through a dataset composed of various types of produce, ran through a single program.</h4> 

<ul>
  <li>Multi-threaded build utilizes a GPU-accelerated framework using Nvidia CUDA technology in addition to regular CPU multi-threading</li>
  <li>Benchmarks each framework's performance through multiple data subsets and sizes​</li>
  <li>Compares test results of single vs. multithreaded ​</li>
</ul>

<br><br>

## How to run


<h4> 1.)	Download the project from the <a href= "https://github.com/jacobchung02/produceNet" target="_blank">Github</a> and open in any python IDE </h4>
 <br><br>
  <img src="https://github.com/user-attachments/assets/9e1eba95-fd9e-414b-8c06-b884008c76d1" width="500" height="400"/>

<br><br>

<h4>2.)	Download <a href= "https://developer.nvidia.com/cuda-11-8-0-download-archive" target="_blank">CUDA Toolkit 11.8</a> to utilize CUDA GPU acceleration </h4>
<br><br>
  <img src="https://github.com/user-attachments/assets/48a4ff56-4bb3-4cfd-90fb-4f055715174e" width="1100" height="300"/>
<br><br>

<h4>3.)	Download PyTorch and its companion libraries using (pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118)</h4>

<br><br>

<h4>4.)	Depending on your CPU type, uncomment line #15 if you are running on an intel CPU</h4>

<br><br>

<img src="https://github.com/user-attachments/assets/f3ea33d8-8053-488f-9419-674bc1ebb537" width="1100" height="75"/>

<br><br>

<h4>5.)	Once this is all set, you are ready to run the code</h4>

<img src="https://github.com/user-attachments/assets/9bf2bc68-3bd5-44a6-9d3f-feb761219ce0" width="600" height="175"/>
<br><br>

## Functions

<h3><b></b>Load Function: load_kaggle_datasets() </b></h3>
<img src="https://github.com/user-attachments/assets/c894649a-ba78-495a-bf24-7786931bb035"/>
<h4> In this function, we utilized DataLoader, which is a PyTorch tool that allows for parallelization, in our loading scheme.  </h4>
<h4> The purpose of this load function was to make an efficient loading scheme that includes parallelization loading, from our image dataset, that is located in a directory in the root. </h4>

<br><br>

<h3><b></b>Benchmark Tests</b></h3>
<img src="https://github.com/user-attachments/assets/39a1db3e-d4e9-478b-aba8-e42afa0c22bf"/>

<h4> This is where our code for the convolutional neural network (CNN) runs tests on each model of thread type between CPU and GPU. </h4>
<h4>Our outer loop calls from the Loader functions to create an iteration over different batch sizes. </h4>
<h4>CNN training is done on the CPU with different thread counts. The first training will be done single threaded, then with two threads, then four threads, then finally eight threads. </h4>
<h4>CNN training is also done on the GPU across the amount of threads that are present on your GPU.  </h4>
<h4>The following metrics are measured. </h4>
<ul>
  <li>Execution Time </li>
  <li>Accuracy </li>
  <li>Batch Size </li>
</ul>

<br><br>

## Output
<h4>At the end, your output will include the following:</h4>

<h4>Console output of each model benchmark</h4>
<img src="https://github.com/user-attachments/assets/0756906e-5625-4ddb-96f4-6f15b9a0b5f6" /img>
<h4>One bar graph of execution​ time, per batch</h4>
<img src="https://github.com/user-attachments/assets/caa24ca0-f36a-4b86-8ea2-2e76b1cfb0cd"/img>
<h4>One line graph of validation accuracy history over time​, per batch</h4>
<img src="https://github.com/user-attachments/assets/0d163c18-2262-4c24-8e92-61c1c0d60d78"/img>
<h4>Graphs will be stored locally on your machine in memory as part of the matplotlib sidebar</h4>
<img src="https://github.com/user-attachments/assets/6c9b3ce5-30f2-4053-8075-d1c17bbb06f4"/img>




