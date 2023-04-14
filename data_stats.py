'''
Updated: 2022-11-28
Refactored plot code into show_plot().

Takes dataset information (age, gender, ethnicity) and displays
each distribution as a bar chart.
'''
import matplotlib.pyplot as plt
from os import walk

# Relative dir path of dataset images
# Image filename in dataset folder must formatted "<age>_<gender>_<ethnicity>_*"
dataset_folder = "dataset"


'''
Parameters:
    data -> dict(<string>: <int>)
    title -> string

Function:
    Uses pyplot to display the "data" dictionary as a bar chart with "title".
'''
def show_plot(data, title):
    plt.figure(figsize=(12,6))
    plt.bar(range(len(data)), list(data.values()), align='center')
    plt.xticks(range(len(data)), list(data.keys()))
    plt.title(title)
    
    plt.show()


def main():
    """
    # A description of the dataset can be found here:
    # https://susanqq.github.io/UTKFace/

    You need to count the number of instances that each age, each gender, and
    each ethnicity occurs in the dataset. Store that info in the below
    dictionaries as appropriate (e.g. {18: 284, 19: 271} might be two appropriate
    entries in the "ages" dict, {"Male": 8419} might be an entry in "genders", etc.)
    You'll need to convert ethnicities from numbers to strings!
    """
    
    ages = dict()
    genders = dict()
    ethnicities = dict()
    
    # TODO: Walk through the dataset folder and fill the dictionaries...
    raise NotImplementedError("Hint: use os.walk")

    """
    You don't need to edit below here. Basically, each time you call show_plot,
    a window will pop up with the appropriate graph. When you close the window,
    the next show_plot will pop up. Once you've closed all three, the program
    will terminate.
    """
    
    # Plot age graph and show
    show_plot(ages, "Age")
    
    # Plot gender graph and show
    show_plot(genders, "Gender")
    
    # Plot ethnicity graph and show
    show_plot(ethnicities, "Ethnicity")


if __name__=="__main__":
    main()
