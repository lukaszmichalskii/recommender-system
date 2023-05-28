<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![CI][ci-shield]][ci-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/lukaszmichalskii/recommender-system">
    <img src="docs/images/logo.png" alt="Logo" width="150" height="150">
  </a>

  <h3 align="center">Recommender System</h3>

  <p align="center">
    Application for movies recommendation using machine learning techniques
    <br />
    <a href="https://github.com/lukaszmichalskii/recommender-system#more-about-recommender-system"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/lukaszmichalskii/recommender-system">View Demo</a>
    ·
    <a href="https://github.com/lukaszmichalskii/recommender-system/issues">Report Bug</a>
    ·
    <a href="https://github.com/lukaszmichalskii/recommender-system/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents
- [More About Recommender System](#more-about-recommender-system)
  - [Recommender System](#recommender-system)
  - [Multi-Regression](#multi-regression)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Get the Recommender System source](#get-the-recommender-system-source)
  - [Install Dependencies](#install-dependencies)
- [Getting Started](#getting-started)
- [Continous Integration](#continuous-integration)
- [License](#license)
- [Backlog](#backlog)
- [Contact](#contact)


## More About Recommender System
Recommendation engines are a subclass of machine learning which generally deal with ranking or rating products / users. 
Loosely defined, a recommender system is a system which predicts ratings a user might give to a specific item. 
These predictions will then be ranked and returned back to the user.

Recommender systems are really critical in some industries as 
they can generate a huge amount of income when they are efficient 
or also be a way to stand out significantly from competitors. 
As a proof of the importance of recommender systems,
a few years ago, Netflix organised a challenges (the “Netflix prize”) where 
the goal was to produce a recommender system that performs better than 
its own algorithm with a prize of 1 million dollars to win.

Implemented system can be easily integrated into other applications.

### Collaborative filtering based approach
Collaborative methods for recommender systems are methods that are based solely on the past interactions recorded between users and items in order to produce new recommendations. These interactions are stored in the so-called “user-item interactions matrix”.
Then, the main idea that rules collaborative methods is that these past user-item interactions are sufficient to detect similar users and/or similar items and make predictions based on these estimated proximities.

The class of collaborative filtering algorithms is divided into two sub-categories that are generally called memory based and model based approaches. Memory based approaches directly works with values of recorded interactions, assuming no model, and are essentially based on nearest neighbours search (for example, find the closest users from a user of interest and suggest the most popular items among these neighbours). Model based approaches assume an underlying “generative” model that explains the user-item interactions and try to discover it in order to make new predictions.

The main advantage of collaborative approaches is that they require no information about users or items and, so, they can be used in many situations. Moreover, the more users interact with items the more new recommendations become accurate: for a fixed set of users and items, new interactions recorded over time bring new information and make the system more and more effective.

However, as it only consider past interactions to make recommendations, collaborative filtering suffer from the “cold start problem”: it is impossible to recommend anything to new users or to recommend a new item to any users and many users or items have too few interactions to be efficiently handled. This drawback can be addressed in different way: recommending random items to new users or new items to random users (random strategy), recommending popular items to new users or new items to most active users (maximum expectation strategy), recommending a set of various items to new users or a new item to a set of various users (exploratory strategy) or, finally, using a non collaborative method for the early life of the user or the item.

### Multi-Regression
System fit model using multiple regression calculation for each user. 
This calculation can be done offline to ensure robust performance. 
The application maintainer should schedule a nightly job that fetches 
all user's data and fits multi-regression models for each user. 
Thanks to this customers do not see any performance issues because 
we use only a "lookup table" or predicted model utility in the background. 
Moreover, to show items related to the browsed one we can calculate MSE 
(mean-squared error) offline too. Then as mentioned above our product will 
immediately show related items.

## Installation

### Prerequisites
If you are installing from source, you will need:
- Python 3.8 or later

If you want to compile with CUDA support, install the following (note that CUDA is not supported on macOS)
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads) 11.0 or above. Make sure TensorFlow support installed CUDA version.
- [NVIDIA cuDNN](https://developer.nvidia.com/cudnn) v7 or above
- [Compiler](https://gist.github.com/ax3l/9489132) compatible with CUDA

Note: You could refer to the [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Support-Matrix.pdf) for cuDNN versions with the various supported CUDA, CUDA driver and NVIDIA hardware

### Get the Recommender System source
```bash
git clone https://github.com/lukaszmichalskii/recommender-system.git
cd recommender-system
# **** OPTIONAL: virtual environment for Python setup ****
python3 -m virtualenv venv
source venv/bin/activate
# **** END OPTIONAL ****
```

### Install Dependencies
```bash
python3 -m pip install -r build_requirements.txt
# install tensorflow CUDA optimized library, for CPU use `python -m pip install tensorflow-cpu`
python3 -m pip install tensorflow
```


> _Aside:_ You may end up with incorrect CPU based TensorFlow installation caused by pip does not detect CUDA:.
> If you have CUDA enabled just follow instruction from [TensorFlow GPU optimized](https://www.tensorflow.org/guide/gpu) and override installed TensorFlow version.
> 


## Getting Started

Recommender System can be run using rating file provided by actor.

Example run with provided resources, for file structure check: [convention](https://github.com/lukaszmichalskii/recommender-system/blob/master/test/resources/ratings.csv):
```bash
# standard mode
python3 src/app_runner.py --ratings <ratings_file_example>
```

#### Additional data from Google Knowledge Graph API
> _Aside:_ Google Cloud API key for Knowledge Graph service needed.
> 

```bash
export API_KEY=<private_google_cloud_key>
python3 src/app_runner.py --ratings <ratings_file_example>
```

#### More information during execution
If you want to see more information run enable `--verbose` flag. To specify output directory
use `--output <path_to_dir>` option (default save to `results/` directory).
```bash
# save results from run
python3 src/app_runner.py --ratings <ratings_file_example> --verbose
```

#### Adjust running options
Other options can be set via environment variables:

| Variable              | Description                                                                                                                                                                                                                                                                                                                                                | Default                                                                                                   |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| RECOMMENDATIONS_LIMIT | Specifies how much recommendation should be presented                                                                                                                                                                                                                                                                                                      | 10                                                                                                        |
| PRECISION             | Recommendations predictions metric. Enable to customize model performance.  High PRECISION value provide more accurate recommendations but require longer execution time. Low PRECISION value provide less accurate recommendations with faster execution time. Default precision was selected based on a lot of analysis and optimal value was determined | 200                                                                                                       |
| CPU_THREADS           | Specifies how many cores should be utilized during application execution.                                                                                                                                                                                                                                                                                  | Use hyper-threading by count physical cores and threads that could be executed in parallel on single core |

## Continuous Integration
Source code is maintained using GitHub Actions CI job. Each changes, pushed to repository are validated by linters check and test job.
#### Code style
`pylint`, `black`

## License
Recommender System has a MIT license, as found in the [LICENSE](https://github.com/lukaszmichalskii/camera-perception/blob/master/LICENSE) file.


<!-- BACKLOG -->
## Backlog

See the [Issues](https://github.com/lukaszmichalskii/recommender-system/issues) for a list of proposed features (and known issues).


<!-- CONTACT -->
## Contact

Project: [https://github.com/lukaszmichalskii/recommender-system](https://github.com/lukaszmichalskii/recommender-system)

| Author       | Email        |
|--------------|--------------|
| Łukasz Michalski | 261118@student.pwr.edu.pl |


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/lukaszmichalskii/repo.svg?style=flat-square
[contributors-url]: https://github.com/lukaszmichalskii/recommender-system/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/lukaszmichalskii/repo.svg?style=flat-square
[forks-url]: https://github.com/lukaszmichalskii/recommender-system/network/members
[stars-shield]: https://img.shields.io/github/stars/lukaszmichalskii/repo.svg?style=flat-square
[stars-url]: https://github.com/lukaszmichalskii/recommender-system/stargazers
[issues-shield]: https://img.shields.io/github/issues/lukaszmichalskii/repo.svg?style=flat-square
[issues-url]: https://github.com/lukaszmichalskii/recommender-system/issues
[license-shield]: https://img.shields.io/badge/license-MIT-orange
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/lukasz-michalski-823106202/
[jira-shield]: https://img.shields.io/badge/Jira-Join-blue
[ci-shield]: https://img.shields.io/badge/CI-passing-green
[ci-url]: https://github.com/lukaszmichalskii/recommender-system/actions/