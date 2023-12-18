"""
Updated: 2022-11-28
Refactored plot code into show_plot().

Takes dataset information (age, gender, ethnicity) and displays
each distribution as a bar chart.

Written by Steve Legere for OPC/Technology Analysis Directorate
"""
import matplotlib.pyplot as plt
from os import walk

# Relative dir path of dataset images
# Image filename in dataset folder must formatted "<age>_<gender>_<ethnicity>_*"
dataset_folder = "dataset"


"""
Uses pyplot to display the "data" dictionary as a bar chart with "title".
"""
def show_plot(data, title):
    plt.figure(figsize=(12,6))
    plt.bar(range(len(data)), list(data.values()), align='center')
    plt.xticks(range(len(data)), list(data.keys()))
    plt.title(title)
    
    plt.show()


def main():
    """
    A description of the dataset can be found here:
    https://susanqq.github.io/UTKFace/
    """
    
    ages = dict()
    genders = dict()
    ethnicities = dict()
    
    # Walk through the dataset folder and fill the dictionaries...
    for path, dirs, files in walk(dataset_folder):
        for file in files:
            # Get metadata from filenames
            meta = file.split("_")
            a, g, e = int(meta[0]), int(meta[1]), int(meta[2])

            # Place metadata into each dictionary
            ages[a] = ages[a] + 1 if a in ages.keys() else 1
            genders[g] = genders[g] + 1 if g in genders.keys() else 1
            ethnicities[e] = ethnicities[e] + 1 if e in ethnicities.keys() else 1
        break
    
    # Put descriptive labels in the dictionaries
    try:
        genders["Male"] = genders.pop(0)
        genders["Female"] = genders.pop(1)
        
        ethnicities["White"] = ethnicities.pop(0)
        ethnicities["Black"] = ethnicities.pop(1)
        ethnicities["Asian"] = ethnicities.pop(2)
        ethnicities["India"] = ethnicities.pop(3)
        ethnicities["Other"] = ethnicities.pop(4)
        
    except KeyError as e:
        print(f'Warning: skipped key/value pair (not found): {e}')
    
    # Plot age graph and show
    show_plot(ages, "Age")
    
    # Plot gender graph and show
    show_plot(genders, "Gender")
    
    # Plot ethnicity graph and show
    show_plot(ethnicities, "Ethnicity")

if __name__ == "__main__":
    main()
