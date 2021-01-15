<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">MLProject</h3>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Installation](#installation)
* [Usage](#usage)
* [Parameters](#parameters)
* [Sources](#sources)

<!-- ABOUT THE PROJECT -->
## About The Project

### Built With

* [Anaconda](https://www.anaconda.com/)
* [mlFlow](https://mlflow.org/)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Installation

1. Clone the repo

    ```sh
    git clone https://github.com/Twizzle1997/MLProject
    ```
    
2. Create a conda virtual environment with

    ```sh
    conda env create -f src/conda.yaml
    ```

<!-- USAGE EXAMPLES -->
## Usage

* Run in terminal ```python notebook.py {kernel} {stride} {epochs}```  
Example :
    ```sh
    python notebook.py 3 2 5
    ```

* Run in terminal ```mlflow ui```  
* Launch [http://localhost:5000/](http://localhost:5000/)

## Parameters
```{kernel}``` (type:int, default:3) Kernel size  
```stride``` (type:int, default:1) Strides size  
```epochs``` (type:int, default:15) Number of epoch  


## Sources

* https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html
