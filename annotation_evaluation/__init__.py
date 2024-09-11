import os
import pandas as pd

CATEGORIES = {
    "Religion": {
        "Christian": ["Scumbag-Catholic-Priest", "Condescending-Christian", "Jesus-Christ"],
        "Islam": ["Angry Muslim Guy", "Muslim-Immigrant", "Confused-Muslim-Girl"],
        "Judaism": ["Advicejew", "Jewish-Dude", "Like-A-Jew"]
    },

    "Countries": {
        "Germany": ["Typical-Germany-Lover", "Germany-Pls", "Success-Germany"],
        "USA": ["American Pride Eagle", "American-Flag-Shotgun-Guy", "Obese-American"],
        "Mexico": ["Successful Mexican", "Mexicanotriste", "Mexicans-On-A-Truck"],
        "China": ["Stern-But-Honest-Chinese-Guy", "Good-Chinese-Student", "Nia-China"],
        "India": ["Generic Indian Guy", "Indian-Father", "Stereotypical-Indian-Telemarketer"]
    },

    "Ethnicity": {
        "Asian": ["Asian-College-Freshman", "Bad-Advice-Asian", "Troll-Asian"],
        "Black": ["Sassy Black Woman", "Black Kid", "Skeptical-Black-Kid"],
        "Middle Eastern": ["Y-U-So-Arab", "Arabic-Meme", "Richarabclap"],
        "White": ["Privilege-Abusing-White-Couple", "Nice-White-Girl", "White-Power-Problems"]
    },

    "Sexual Orientation and Gender Identity": {
        "Trans": ["Oppressive-Trans-Bro", "Privilege-Denying-Tranny", "Transvestite-Trevor"]
    },

    "Cultural and Political Topics": {
        "Gender and Family": ["Feminist Cunt", "Privilege-Denying-Feminist", "Feministfrequently"],
        "Police Man": ["Scumbag-Police-Officer", "Strict-Policeman", "Policeman"]
    }
}


path_folder = "/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/dataset/annotation/hatespeech_nonhate/images/en"

mapping = {}
# Iterate through all directories and files in the given folder
for root, dirs, files in os.walk(path_folder):
    template_name = root.split("/")[-1].lower()
    mapping[template_name] = []
    for filename in files:
        if ".jpg" in filename:
            filename = filename.split(".")[0]
            mapping[template_name].append(filename)

# Converting the dictionary to a pandas DataFrame
rows = []
for category, subcategories in CATEGORIES.items():
    for subcategory, items in subcategories.items():
        for item in items:
            item = item.replace(" ", "-")
            for instances in mapping[item.lower()]:
                rows.append([category, subcategory, item, instances])
df = pd.DataFrame(
    rows, columns=["Category", "Subcategory", "Template", "Instance"])
df.to_csv('/Users/duc/Desktop/Projects/Ongoing/MultiModalMemes/annotation_evaluation/category_mapping.csv', index=False)
