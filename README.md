# UrbanAIZE

Large amounts of map sheets and drawings are archived in society as scanned raster images or paper maps (that could be scanned). Examples include zoning maps or property drawings. These are often used as reference materials in urban planning or renovations. However, in order to use them as reference materials, they need to be digitized, which can be a time-consuming manual task. This project aims to automate the process that would facilitate the digitization and make old map data more easily accessible and reusable.

_More information in Swedish):_ [https://www.oru.se/samverkan/oru-innovation/ai-impact-lab/avslutade-projekt/urbanaize/](https://www.oru.se/samverkan/oru-innovation/ai-impact-lab/avslutade-projekt/urbanaize/)

_Web demo as a result of this project:_ [https://urbanaize.aass.oru.se/](https://urbanaize.aass.oru.se/)


## Background and Purpose

In 2020, [Statistics Sweden (SCB)](https://www.scb.se/) scanned approx. 8,000 map sheets from the Population and Housing Censuses (FoB) in 1975 and 1980, respectively. Most of the map sheets are black-and-white economic sheets with delineation lines drawn in color for urban areas that were applied at the respective time. The map sheets are in the raster format TIF and sorted into folders based on county and municipality affiliation. An example of such a map image can be seen in Image 1. The resolution for each map sheet is 300 dpi with approx. 7500 x 7000 pixels. The demarcations for the urban areas can be linked to statistics and contribute to a geographical description both of the rapid urbanization in connection with the million program and the breaking point of the so-called green wave. The material is, therefore, deemed to contain valuable information and relevant to digitize, to be then made available as open data.

![Exempel på kartbild](./images/example.png)
<sup align="center"><b>Image 1:</b> An example of a scanned map sheet as a result of the Population and Housing Censuses (FoB) in 1975. .<sup>  

In this project, it is mainly the borders in FoB 1975 that are of interest. These borders have previously not been digitized and made available as vectorized geographical coordinates. Consequently, it is not possible to fully follow and understand Swedish urbanization and the geographical development of urbanization, which the scanned map material will be able to contribute to. The project aims to make available map material that otherwise risks being left unused in archives due to time-consuming manual digitization processes. Many actors in society would benefit from an automated digitization process as archived maps and drawings would become more accessible to community planners, construction companies, architects, individuals, and researchers alike, which could all benefit from accessible historical geographic data.


## Project Structure

This project assumes the following project structure:

     UrbanAIZE/
     ├── images/                        # Images used for visualization
     ├── scripts/                  		
         ├── annotation.py              # Annotation of map images
         ├── dataset.py                 # Handles image pairs during training 
         ├── __init__.py
         ├── map_image.py               # Helper class used for processing map image
         └── model/
             ├── block.py               # Common network block layers
             ├── unet.py                # Implementation of the classic U-Net model
             └── unet2plus.py           # Implementation of the U-Net++ model
         ├── preparation.py             # Preperation of map images
         ├── train.py                   # Traning of machine learning models	
         ├── uitls.py                   # General util functions
         └── window_handler.py          # Handles display of map images
     ├── requirements.txt               # Required Python libraries
     ├── .gitginore              
     ├── LICENSE                        # MIT License
     └── README   


## Installation

1. Clone this GitHub repository:       
        
        git clone https://github.com/statisticssweden/UrbanAIZE.git 
       
2. Install required Python libraries:
        
        cd UrbanAIZE && pip3 install -r requirements.txt
        
## Operation

As an initial procedure, download map images (in `.tiff` format), including the database file `Tatorter_1980_2020.gpkg`, and place them in a local folder. This folder's default name and location is `./data`, but the folder's path and name can also be specified by the optional `--path` argument for all the operations listed below. 
 
### 1. Preperation

	python3 scripts/preperation.py [--path <path>]
	
### 2. Annotation

	python3 scripts/annotation.py [--path <path>]

### 3. Training

	python3 scripts/train.py [--path <path>]
	
### 4. Prediction
	
	python3 scripts/prediction.py [--path <path>]
	
