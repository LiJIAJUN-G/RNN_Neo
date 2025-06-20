<h1 align="center"> RNN_Neo </h1>

<h6 align="center"> An RNN model framework, simulates the processing of neo-peptides in vivo from proteasome cleavage, TAP transport, pMHC binding to TCR activation. </h6>  

<p align="center"> 
  <img src="./fig/Signal.gif" alt="Sample signal" width="70%" height="70%">
</p>


## :book: Install
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


#### Dependencies
RNN_Neo currently test on x86_64 on ubuntu 16.04 with 96-core cpu


#### Get the RNN-Neo Source

```
git clone https://github.com/LiJIAJUN-G/RNN_Neo.git
```
#### Create and activate an environment using conda

```
conda env create -f freeze.yml
conda activate rnn_neo
```

#### Download Reference genome

```
bash download_hg38.sh
```

:cactus:Make sure the `ref` directory is as follows:
```
ref
├── anno
│   ├── 1000G_phase1.snps.high_confidence.hg38.vcf.gz(1,844,006kb)
│   ├── 1000G_phase1.snps.high_confidence.hg38.vcf.gz.tbi(2,079kb)
│   ├── af-only-gnomad.hg38.vcf.gz(3,109,644kb)
│   ├── af-only-gnomad.hg38.vcf.gz.tbi(2,386kb)
│   ├── dbsnp_138.hg38.vcf.gz(1,524,307kb)
│   ├── dbsnp_138.hg38.vcf.gz.tbi(2,268kb)
│   ├── Mills_and_1000G_gold_standard.indels.hg38.vcf.gz(20,202kb)
│   ├── Mills_and_1000G_gold_standard.indels.hg38.vcf.gz.tbi(1,465kb)
│   ├── small_exac_common_3.hg38.vcf.gz(1,267kb)
│   └── small_exac_common_3.hg38.vcf.gz.tbi(237kb)
├── hg38.dict(84kb)
├── hg38.fa(3,268,032kb)
├── hg38.fa.amb(22kb)
├── hg38.fa.ann(33kb)
├── hg38.fa.bwt(3,214,439kb)
├── hg38.fa.fai(24kb)
├── hg38.fa.pac(803,610kb)
├── hg38.fa.sa(1,607,220kb)
├── hg38.gtf(1,532,946kb)
├── Homo_sapiens.GRCh38.cdna.all.fa.gz(76,700kb)
├── Homo_sapiens.GRCh38.cdna.all.index(2,569,256kb)
└── TruSeq3-PE.fa(1kb)
```
:exclamation:**Noted**：The `download_hg38.sh` should be executed within the `rnn_neo` environment. `TruSeq3-PE.fa` is available from the source. Due to network fluctuations, file downloads may be interrupted. Ensure that the file size matches the one stated above.


### :airplane:Software Installation

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

#### Mutation Annotation Tool

* [ANNOVAR](https://annovar.openbioinformatics.org/en/latest/)

After downloading, it is also necessary to build the reference library of `hg38` within the `ANNOVAR` folder.
```
perl annotate_variation.pl -downdb -buildver hg38 -webfrom annovar refGene humandb/
```
:exclamation:**Noted**：If you're not utilizing the original input, it may not be necessary to install the software.
#### Feature Extraction Tool
* [NetMHCpan](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/)
* [NetChop](https://services.healthtech.dtu.dk/services/NetChop-3.1/)
* [NetCTLpan](https://services.healthtech.dtu.dk/services/NetCTLpan-1.1/)
* [MixMHCpred](https://github.com/GfellerLab/MixMHCpred)
* [PRIME](https://github.com/GfellerLab/PRIME)
* [BigMHC](https://github.com/KarchinLab/bigmhc)

Install the aforementioned software and include the paths of other software in the environment variable, excluding BigMHC. Ensure that the software can be directly invoked. Modify the path of BigMHC in the `parameter.json` file.

###### Installation tips
- Ensure that  `tcsh` is installed
- Some tools need to download referenced data
- Some tools need to modify the path and `tmp` folder: NetChop, NetMHCpan, etc.
- Some tools need to be compiled using `g++`, such as: MixMHCpred
- For `BigMHC`, the environment is already set up in the rnn_neo conda environment; just download the source code.
- Tools other than `BigMHC` need to be added to the environment variable
```
export PATH=$PATH:your_path/netMHCpan-4.1
```
**In short**, please ensure you carefully read the `README` provided with the above tools. If you have any questions, feel free to contact us for assistance.


### :mag:  Usage

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

1. Configure the working path by modifying the `parameter.json` file to include the current working path.

2. Running code with instances of three input modes. Please be aware: ensure to adjust the input file and parameters according to the provided sample code.

```
python run.py
```

3. If you wish to test the original data input, execute the code in the `test` directory to download the sample data.
```
bash download_test.sh
```

4. The final result output `patientid _ rank _ RNN.csv` should be located within the `calspace` folder.

```
calspace
├── test1
│   └── test1_rank_RNN.csv
├── test2
│   └── test2_rank_RNN.csv
└── test3
    └── test3_rank_RNN.csv
```

### :scroll: Authors and Contact   

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

JiaJun Li and TianYi Qiu

jiajunli23@m.fudan.edu.cn or qiu.tianyi@zs-hospital.sh.cn
Fudan University, Shanghai, China

![RNN_Neo](./fig/RNN_Neo.png)
