# Janus
⚖️ Option weight model for [Ariadne](https://github.com/wsg-ariadne/ariadne).

Janus is an image classifier based on the VGG-19 model that classifies images into the following classes:

1. `absent` indicating that the option to refuse cookies is not on the interface at all
2. `weighted` indicating that the option to refuse cookies is made less obvious, less visible, or more tedious to select than the option to accept it
3. `even` indicating that the options to accept and refuse cookies appear on the cookie banner and are equally obvious.

This classifier allows [Ariadne](https://github.com/wsg-ariadne/ariadne) to determine whether a website uses deceptive design in the form of weighted options on its cookie banner.

## Usage

### Requirements
Install Python 3.8+ (tested on 3.8.16) and the packages in `requirements.txt` using `pip install -r requirements.txt`.

### Generating the model
Run `python generate.py` to reconstruct the pre-trained model and test on an image indicated in line 16 (`"final-dataset/train/weighted/AFPPopup.png"`). This program should output the class index 0, 1, or 2 corresponding to absent, even, and weighted.

### Using the model
Run `python test.py` to have the program provide an opportunity to input the desired image path and provide more explicit output regarding the result.

## Details on the Model

### Dataset
The dataset used to train this model includes photos selected from the Soe, Norberg, Guribye, and Slakovik made available [here](https://github.com/videoworkflow/cookiepopup). The filtering, classification, and labeling applied for this project were done by the developers of [**ariadne**](https://github.com/wsg-ariadne).

    @inproceedings{10.1145/3419249.3420132,
    author = {Soe, Than Htut and Nordberg, Oda Elise and Guribye, Frode and Slavkovik, Marija},
    title = {Circumvention by Design - Dark Patterns in Cookie Consent for Online News Outlets},
    year = {2020},
    isbn = {9781450375795},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3419249.3420132},
    doi = {10.1145/3419249.3420132},
    abstract = { To ensure that users of online services understand what data are collected and how they are used in algorithmic decision-making, the European Union’s General Data Protection Regulation (GDPR) specifies informed consent as a minimal requirement. For online news outlets consent is commonly elicited through interface design elements in the form of a pop-up. We have manually analyzed 300 data collection consent notices from news outlets that are built to ensure compliance with GDPR. The analysis uncovered a variety of strategies or dark patterns that circumvent the intent of GDPR by design. We further study the presence and variety of these dark patterns in these “cookie consents” and use our observations to specify the concept of dark pattern in the context of consent elicitation.},
    booktitle = {Proceedings of the 11th Nordic Conference on Human-Computer Interaction: Shaping Experiences, Shaping Society},
    articleno = {19},
    numpages = {12},
    keywords = {dark patterns, cookie consent notice, CCPA, GDPR},
    location = {Tallinn, Estonia},
    series = {NordiCHI '20}
    }

### Training
The training can be found in `janus.ipynb` and follows Nithyashree V's Step-by-Step guide for Image Classification on Custom Datasets [here](https://www.analyticsvidhya.com/blog/2021/07/step-by-step-guide-for-image-classification-on-custom-datasets/).

The developers used a VGG-19 model on a custom data set to classify photos of cookie banners to three categories:
    1. `absent` indicating that the option to refuse cookies is not on the interface at all
    2. `weighted` indicating that the option to refuse cookies is made less obvious, less visible, or more tedious to select than the option to accept it
    3. `even` indicating that the options to accept and refuse cookies appear on the cookie banner and are equally obvious.