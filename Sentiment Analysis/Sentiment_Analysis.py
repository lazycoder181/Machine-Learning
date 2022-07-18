'''
The Sentiment Analysis script performs the following tasks:
1. Categorises comments based on keywords.
2. Removes the uncategorised comments
3. Provides sentiments for the categorised comments

Instructions to run the script:
1. Install python 3.6 or above
2. Install any IDE for running this script
3. Since there will be multiple files containing comments, please make sure to merge all the files before providing them as the input
4. Please make sure that your name of the column for comments is "Answer"
5. Categories: Accommodation, Cleanliness, Facilities, Food, Technology, Transport, Utilities, General,	Mandate, Health Concerns, Indigenous, Racism, Culture and Support, Safe Production System, Career Development
6. Uncomment "nltk.download('vader_lexicon')" while executing the script for the first time. This will download the vader_lexicon. If there is a need to run the script again, comment nltk.download('vader_lexicon')
7. Run the script using the run command from your cmd prompt or your IDE's 'run option.

'''

import pandas as pd
# nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Merger file contains all the spreadsheets combined in one
df = pd.read_csv("Comments_Merger .csv")

# df = pd.read_csv("phy_vs_psy.csv")

# Categorizing Comments based on keywords
print("Categorizing comments based on Accommodation\nPlease wait...")
# Keywords for accommodation
acc_list = [" camp ", " village ", " accommodation ", " accomodation ", " acommodation ", " acomodation ",
            " searipple ", " air con ",
            " air conditioning ", " aircon ", " town ", " property ", " residential ", " resi ", " hous ", " sodexo ",
            " sedexo ",
            " dry camp ",
            " sea ripple ", " House ", " Housing ", " room sharing ", " showers ", " transit facility ",
            " refurbishment ",
            " not enough accommodation ", " Barimunya ", " Bolgeeda ", " AC ", " Donga ", " camp lighting "]

for accommodation in acc_list:
    df[accommodation] = df.astype(str).sum(axis=1).str.contains(accommodation)

df["Accommodation"] = df[" camp "] + df[" village "] + df[" accommodation "] + df[" accomodation "] + df[
    " acommodation "] + df[
                          " searipple "] + df[" air con "] + df[" air conditioning "] + df[" aircon "] + df[" town "] + \
                      df[" property "] + df[
                          " residential "] + df[" resi "] + df[" hous "] + df[" sodexo "] + df[" sedexo "] + df[
                          " dry camp "] + df[
                          " sea ripple "] + df[" House "] + df[" Housing "] + df[" room sharing "] + df[" showers "] + \
                      df[
                          " transit facility "] + df[" refurbishment "] + df[" not enough accommodation "] + df[
                          " Barimunya "] + \
                      df[" Bolgeeda "] + df[" AC "] + df[" Donga "] + df[" camp lighting "]

df.drop([" camp ", " village ", " accommodation ", " accomodation ", " acommodation ", " acomodation ", " searipple ",
         " air con ",
         " air conditioning ", " aircon ",
         " town ", " property ", " residential ", " resi ", " hous ", " sodexo ", " sedexo ", " dry camp ",
         " sea ripple ", " House ",
         " Housing ", " room sharing ",
         " showers ", " transit facility ", " refurbishment ", " not enough accommodation ", " Barimunya ",
         " Bolgeeda ", " AC ",
         " Donga ", " camp lighting "], axis=1, inplace=True)

print("Done")

# Keywords for cleanliness
print("Categorizing comments based on Cleanliness\nPlease wait...")
# New keywords = stains, dirt
clean_list = [" clean ", " mold ", " mould ", " dirty ", " stains ", " Sodexo cleaning ", " dirt "]

for cleanliness in clean_list:
    df[cleanliness] = df.astype(str).sum(axis=1).str.contains(cleanliness)

df["Cleanliness"] = df[" clean "] + df[" mold "] + df[" mould "] + df[" dirty "] + df[" stains "] + df[
    " Sodexo cleaning "] + df[
                        " dirt "]

df.drop([" clean ", " mold ", " mould ", " dirty ", " stains ", " Sodexo cleaning ", " dirt "], axis=1, inplace=True)

print("Done")

# print(df.columns)

# Keywords for facilities
print("Categorizing comments based on Facilities\nPlease wait...")
facilities_list = ["facility", "gym", "recreation", "sport"]

for facilities in facilities_list:
    df[facilities] = df.astype(str).sum(axis=1).str.contains(facilities)

df["Facilities"] = df["facility"] + df["gym"] + df["recreation"] + df["sport"]

df.drop(["facility", "gym", "recreation", "sport"], axis=1, inplace=True)

print("Done")

# print(df.columns)

# Keywords for food
print("Categorizing comments based on Food\nPlease wait...")
food_list = [" crib ", " menu ", " food ", " lunch ", " dinner ", " breakfast ", " meal ", " eat ", " cater ",
             " alcohol ", " drink ",
             " booze ", " eating ", " Sodexo catering ", " sodexo ",
             "fruit", " veg ", " veggies ", " vegetables ", " meat "]

for foods in food_list:
    df[foods] = df.astype(str).sum(axis=1).str.contains(foods)

df["Food"] = df[" crib "] + df[" menu "] + df[" food "] + df[" lunch "] + df[" dinner "] + df[" breakfast "] + df[
    " meal "] + df[
                 " cater "] + df[" alcohol "] + df[" drink "] + df[" booze "] + df[" eat "] + df[" eating "] + df[
                 " Sodexo catering "] + df["fruit"] + df[" veg "] + df[" veggies "] + df[" vegetables "] + df[
                 " meat "] + df[" sodexo "]

df.drop([" crib ", " menu ", " food ", " lunch ", " dinner ", " breakfast ", " meal ", " eat ", " cater ", " alcohol ",
         " drink ", " booze ",
         " eating ", " Sodexo catering ",
         "fruit", " veg ", " veggies ", " vegetables ", " meat ", " sodexo "], axis=1, inplace=True)

print("Done")

# print(df.columns)

# Keywords for Technology
print("Categorizing comments based on Technology\nPlease wait...")
tech_list = ["internet", "telstra", "wifi", "wi-fi", "foxtel", " tv "]

for techs in tech_list:
    df[techs] = df.astype(str).sum(axis=1).str.contains(techs)

df["Technology"] = df["internet"] + df["telstra"] + df["wifi"] + df["wi-fi"] + df["foxtel"] + df[" tv "]

df.drop(["internet", "telstra", "wifi", "wi-fi", "foxtel", " tv "], axis=1, inplace=True)
print("Done")

# print(df.columns)

# Keywords for transport
print("Categorizing comments based on Transport\nPlease wait...")
trans_list = ["the bus", "buses", "transport", "airport", "commute", "flight", "fly", "Flying", " DIDO ",
              "travel hours",
              "travel",
              "travel time", "WA airport", "Karratha Airport", "Perth Airport"]

for transport in trans_list:
    df[transport] = df.astype(str).sum(axis=1).str.contains(transport)

df["Transport"] = df["the bus"] + df["buses"] + df["transport"] + df["airport"] + df["commute"] + df["flight"] + df[
    "fly"] + df["Flying"] + df[" DIDO "] + df["travel hours"] + df["travel"] + df["travel time"] + df["WA airport"] + \
                  df[
                      "Karratha Airport"] + df["Perth Airport"]

df.drop(
    ["the bus", "buses", "transport", "airport", "commute", "flight", "fly", "Flying", " DIDO ", "travel hours",
     "travel",
     "travel time", "WA airport", "Karratha Airport", "Perth Airport"], axis=1, inplace=True)

print("Done")

# Keywords for Utilities
print("Categorizing comments based on Utilities\nPlease wait...")
utils_list = ["cost of living", "living cost", "electric", "subsidise", "subsidies", "subsidize", "subsid", "utilit",
              " bill ", "water", "Utility"]

for utilities in utils_list:
    df[utilities] = df.astype(str).sum(axis=1).str.contains(utilities)

df["Utilities"] = df["cost of living"] + df["living cost"] + df["electric"] + df["subsidise"] + df["subsidies"] + df[
    "subsidize"] + df["subsid"] + df["utilit"] + df[" bill "] + df["water"] + df["Utility"]

df.drop(
    ["cost of living", "living cost", "electric", "subsidise", "subsidies", "subsidize", "subsid", "utilit", " bill ",
     "water", "Utility"],
    axis=1, inplace=True)

print("Done")

# Keywords for General
print("Categorizing comments based on General\nPlease wait...")
general_list = ["covid", "covid-19", " sars ", "morbidit", "cov-2", "transmission", "delta", "lambda", "disease",
                "cov19",
                "coronavirus", "virus", "bat virus",
                "valued", "belong", "heard", "cared", "care", "value", "seek", "safe", "change"]

for general in general_list:
    df[general] = df.astype(str).sum(axis=1).str.contains(general)

df["General"] = df["covid"] + df["covid-19"] + df[" sars "] + df["morbidit"] + df["cov-2"] + df["transmission"] + df[
    "delta"] + df["lambda"] + df["disease"] + df["cov19"] + df["coronavirus"] + df["virus"] + df["bat virus"] + df[
                    "valued"] + df["belong"] + df["heard"] + df["cared"] + df["care"] + df["value"] + df["seek"] + df[
                    "safe"] + df["change"]

df.drop(["covid", "covid-19", " sars ", "morbidit", "cov-2", "transmission", "delta", "lambda", "disease", "cov19",
         "coronavirus", "virus", "bat virus",
         "valued", "belong", "heard", "cared", "care", "value", "seek", "safe", "change"], axis=1, inplace=True)

print("Done")

# Keywords for Mandate
print("Categorizing comments based on Mandate\nPlease wait...")

mandate_list = ["pfizer", "astra zeneca", "astrazeneca", "moderna", "vaccinat", "vaccine", "fizer", "forced", "The UN",
                "mandate", "mandatory", "public health", "betrayed", "informed consent", "cdc", "legislation",
                "bioethic", "anti-vax", "freedom", "awake", "injected", "ideology", "disinformation", "misinformation",
                "pro-choice", "prochoice", "communism", "communist", "tyranny", "coercion", "free choice", "protest",
                "dictat"]

for mandate in mandate_list:
    df[mandate] = df.astype(str).sum(axis=1).str.contains(mandate)

df["Mandate"] = df["pfizer"] + df["astra zeneca"] + df["astrazeneca"] + df["moderna"] + df["vaccinat"] + df["vaccine"] + \
                df["fizer"] + df["forced"] + df["The UN"] + df["mandate"] + df["mandatory"] + df["public health"] + df[
                    "betrayed"] + df["informed consent"] + df["cdc"] + df["legislation"] + df["bioethic"] + df[
                    "anti-vax"] + df["freedom"] + df["awake"] + df["injected"] + df["ideology"] + df["disinformation"] + \
                df["misinformation"] + df["pro-choice"] + df["prochoice"] + df["communism"] + df["communist"] + df[
                    "tyranny"] + df["coercion"] + df["free choice"] + df["protest"] + df["dictat"]

df.drop(
    ["pfizer", "astra zeneca", "astrazeneca", "moderna", "vaccinat", "vaccine", "fizer", "forced", "The UN", "mandate",
     "mandatory", "public health", "betrayed", "informed consent", "cdc", "legislation", "bioethic", "anti-vax",
     "freedom", "awake", "injected", "ideology", "disinformation", "misinformation", "pro-choice", "prochoice",
     "communism", "communist", "tyranny", "coercion", "free choice", "protest", "dictat"], axis=1, inplace=True)

print("Done")

# Keywords for Health Concerns
print("Categorizing comments based on Health Concerns\nPlease wait...")
hc_list = ["experiment", "experimental", "side effects", "pneumonia", "inject", "proof", "deteriorate",
           "health guideline", "the flu", "a cold", "heart attack", "myocarditis", "blood clots"]

for health_concerns in hc_list:
    df[health_concerns] = df.astype(str).sum(axis=1).str.contains(health_concerns)

df["Health Concerns"] = df["experiment"] + df["experimental"] + df["side effects"] + df["pneumonia"] + df["inject"] + \
                        df["proof"] + df["deteriorate"] + df["health guideline"] + df["the flu"] + df["a cold"] + df[
                            "heart attack"] + df["myocarditis"] + df["blood clots"]

df.drop(
    ["experiment", "experimental", "side effects", "pneumonia", "inject", "proof", "deteriorate", "health guideline",
     "the flu", "a cold", "heart attack", "myocarditis", "blood clots"], axis=1, inplace=True)

print("Done")
# df.to_csv('sample_preprocessed.csv')

# Keywords for Indigenous
print("Categorizing comments based on Indigenous Keywords\nPlease wait...")
ind_list = ["first nations", "original owner", "cultural awareness", "traditional owner", "welcome to country",
            "indigenous", "aborig", "aboriginal", "mob", "fella", "keeping place", "smoking ceremony", "songline",
            "dreamtime", "atsi", "pap", "blak", "Torres", "koori", "noongar", "murri", "custodian", "elder", "lore",
            "legend", "white australian", "pto", "kin", "tiwi", "Eastern Guruma", "Ngarlawangga", "Wiradjuri",
            "Thanikwithi", "Alngith", "Banjima", "Taepadhighi", "Yindjibarndi", "Wik", "Wik-Way", "Nyiyaparli",
            "Kuruma", "Marthudunera", "Ngarluma", "Yinhawangka", "ATAS", "Traditional Owners",
            "Pilbara Aboriginal People"]
for indigenous in ind_list:
    df[indigenous] = df.astype(str).sum(axis=1).str.contains(indigenous)

df["Indigenous"] = df["first nations"] + df["original owner"] + df["cultural awareness"] + df["traditional owner"] + df[
    "welcome to country"] + df["indigenous"] + df["aborig"] + df["aboriginal"] + df["mob"] + df["fella"] + df[
                       "keeping place"] + df["smoking ceremony"] + df["songline"] + df["dreamtime"] + df["atsi"] + df[
                       "pap"] + df["blak"] + df["Torres"] + df["koori"] + df["noongar"] + df["murri"] + df[
                       "custodian"] + df["elder"] + df["lore"] + df["legend"] + df["white australian"] + df["pto"] + df[
                       "kin"] + df["tiwi"] + df["Eastern Guruma"] + df["Ngarlawangga"] + df["Wiradjuri"] + df[
                       "Thanikwithi"] + df["Alngith"] + df["Banjima"] + df["Taepadhighi"] + df["Yindjibarndi"] + df[
                       "Wik"] + df["Wik-Way"] + df["Nyiyaparli"] + df["Kuruma"] + df["Marthudunera"] + df["Ngarluma"] + \
                   df["Yinhawangka"] + df["ATAS"] + df["Traditional Owners"] + df["Pilbara Aboriginal People"]

df.drop(
    ["first nations", "original owner", "cultural awareness", "traditional owner", "welcome to country", "indigenous",
     "aborig", "aboriginal", "mob", "fella", "keeping place", "smoking ceremony", "songline", "dreamtime", "atsi",
     "pap", "blak", "Torres", "koori", "noongar", "murri", "custodian", "elder", "lore", "legend", "white australian",
     "pto", "kin", "tiwi", "Eastern Guruma", "Ngarlawangga", "Wiradjuri", "Thanikwithi", "Alngith", "Banjima",
     "Taepadhighi", "Yindjibarndi", "Wik", "Wik-Way", "Nyiyaparli", "Kuruma", "Marthudunera", "Ngarluma", "Yinhawangka",
     "ATAS", "Traditional Owners", "Pilbara Aboriginal People"], axis=1, inplace=True)

print("Done")

# df.to_csv('sample_preprocessed.csv')

# keywords for Jukkan
print("Categorizing comments based on Jukkan Keywords\nPlease wait...")
jukkan_list = ["aboriginal site",
               "aboriginal sight",
               "sacred site",
               "caves",
               "gorge",
               "Juken",
               "Jukan",
               "Jukkan",
               "Juuken",
               "Juukan",
               "heritage",
               "PKKP",
               "puutu",
               "kunti",
               "kurrama",
               "pinikura",
               "remedial",
               "mediation",
               "remediation",
               "rock shelter",
               "gag clause",
               "Yamatji",
               "Marlpa",
               "ymac",
               "native title",
               "TO Relationship",
               "Pilbara TO",
               "destroyed",
               "Jurken",
               "tragedy"
               ]

for jukkan in jukkan_list:
    df[jukkan] = df.astype(str).sum(axis=1).str.contains(jukkan)

df["Jukkan"] = df["aboriginal site"] + df["aboriginal sight"] + df["sacred site"] + df["caves"] + df["gorge"] + df[
    "Juken"] + df["Jukan"] + df["Jukkan"] + df["Juuken"] + df["Juukan"] + df["heritage"] + df["PKKP"] + df["puutu"] + \
               df["kunti"] + df["kurrama"] + df["pinikura"] + df["remedial"] + df["mediation"] + df["remediation"] + df[
                   "rock shelter"] + df["gag clause"] + df["Yamatji"] + df["Marlpa"] + df["ymac"] + df["native title"] + \
               df["TO Relationship"] + df["Pilbara TO"] + df["destroyed"] + df["Jurken"] + df["tragedy"]

df.drop(["aboriginal site",
         "aboriginal sight",
         "sacred site",
         "caves",
         "gorge",
         "Juken",
         "Jukan",
         "Jukkan",
         "Juuken",
         "Juukan",
         "heritage",
         "PKKP",
         "puutu",
         "kunti",
         "kurrama",
         "pinikura",
         "remedial",
         "mediation",
         "remediation",
         "rock shelter",
         "gag clause",
         "Yamatji",
         "Marlpa",
         "ymac",
         "native title",
         "TO Relationship",
         "Pilbara TO",
         "destroyed",
         "Jurken",
         "tragedy"
         ], axis=1, inplace=True)

print("Done")

# Keywords for Racism
print("Categorizing comments based on Racism\nPlease wait...")
racism_list = ["black lives", "racial", "race", "skin colour", "racism", "racists"]

for race in racism_list:
    df[race] = df.astype(str).sum(axis=1).str.contains(race)

df["Racism"] = df["black lives"] + df["racial"] + df["race"] + df["skin colour"] + df["racism"] + df["racists"]

df.drop(["black lives", "racial", "race", "skin colour", "racism", "racists"], axis=1, inplace=True)

print("Done")

# Keywords for Culture and Support
print("Categorizing comments based on Culture and Support\nPlease wait...")
cul_list = ["Cultural support", "Cultural safety", "Cultural awareness", "Indigenous employment", "Leading Aboriginal",
            "indigenous transfers", "Cultural leave", "Indigenous development",
            "Aboriginal development", "Developing Indigenous", "Developing Aboriginal", "Supporting Indigenous",
            "Supporting Aboriginal"]

for culture in cul_list:
    df[culture] = df.astype(str).sum(axis=1).str.contains(culture)

df["Culture and Support"] = df["Cultural support"] + df["Cultural safety"] + df["Cultural awareness"] + df[
    "Indigenous employment"] + df["Leading Aboriginal"] + df["indigenous transfers"] + df["Cultural leave"] + df[
                                "Indigenous development"] + df["Aboriginal development"] + df["Developing Indigenous"] + \
                            df["Developing Aboriginal"] + df["Supporting Indigenous"] + df["Supporting Aboriginal"]

df.drop(["Cultural support", "Cultural safety", "Cultural awareness", "Indigenous employment", "Leading Aboriginal",
         "indigenous transfers", "Cultural leave", "Indigenous development",
         "Aboriginal development", "Developing Indigenous", "Developing Aboriginal", "Supporting Indigenous",
         "Supporting Aboriginal"], axis=1, inplace=True)

print("Done")

# Keywords for SPS
print("Categorizing comments based on Safe Production System\nPlease wait...")
sps_list = ["SPS",
            "RTSPS",
            "Safe Production Systems",
            "Mindset & Behaviors",
            "M&B",
            "Mindset and Behaviors",
            "M&B",
            "Coaches",
            "lighthouse",
            "mindset",
            "Mirror walk",
            "FLIR",
            "First Line Irritants",
            "Flow",
            "Lean",
            "Change partners",
            "Voyager",
            "Mobius"
            ]

for sps in sps_list:
    df[sps] = df.astype(str).sum(axis=1).str.contains(sps)

df["Safe Production System"] = df["SPS"] + df["RTSPS"] + df["Safe Production Systems"] + df["Mindset & Behaviors"] + df[
    "M&B"] + df["Mindset and Behaviors"] + df["M&B"] + df["Coaches"] + df["lighthouse"] + df["mindset"] + df[
                                   "Mirror walk"] + df["FLIR"] + df["First Line Irritants"] + df["Flow"] + df["Lean"] + \
                               df["Change partners"] + df["Voyager"] + df["Mobius"]

df.drop(["SPS",
         "RTSPS",
         "Safe Production Systems",
         "Mindset & Behaviors",
         "M&B",
         "Mindset and Behaviors",
         "M&B",
         "Coaches",
         "lighthouse",
         "mindset",
         "Mirror walk",
         "FLIR",
         "First Line Irritants",
         "Flow",
         "Lean",
         "Change partners",
         "Voyager",
         "Mobius"
         ], axis=1, inplace=True)

print("Done")

# Keywords for Career Development
print("Categorizing comments based on Career Development\nPlease wait...")
cd_list = ["Training", "OJT", "induction", "onboarding", "new starter", "apprentice", "graduate"]

for cd in cd_list:
    df[cd] = df.astype(str).sum(axis=1).str.contains(cd)

df["Career Development"] = df["Training"] + df["OJT"] + df["induction"] + df["onboarding"] + df["new starter"] + df[
    "apprentice"] + df["graduate"]

df.drop(["Training", "OJT", "induction", "onboarding", "new starter", "apprentice", "graduate"], axis=1, inplace=True)

print("Done")

# Workplace safety
# Psychological
'''
key: 

psy: respect, threat, threatened, threatening, belittling, lesbian, transgender, depression, power imbalance, derogatory, gestures, offensive (done)


physical: violence, assault, injury, harm, hurt, 

'''

print("Categorizing comments based on Psychological Concerns\nPlease wait...")
psy_list = ["safe", " dont feel safe ", " don't feel safe ", " do not feel safe ", " not safe ", "discrimination",
            "discriminatory language", "bad comments", "insult", "uncomfortable", "LGBTQ+", "gay",
            "psychological safety", "bullying", "harassment", "sexual", "sex", "respect", "threat", "threatened",
            "threatening", "belittling", "lesbian", "transgender", "depression", "power imbalance", "derogatory",
            "gestures", "offensive"]
for psy in psy_list:
    df[psy] = df.astype(str).sum(axis=1).str.contains(psy)

df["Psychological H.C"] = df["safe"] + df[" dont feel safe "] + df[" don't feel safe "] + df[
    " do not feel safe "] + df[" not safe "] + df["discrimination"] + df["discriminatory language"] + df[
                              "bad comments"] + \
                          df["insult"] + df["uncomfortable"] + df["LGBTQ+"] + df["gay"] + df["psychological safety"] + \
                          df["bullying"] + df["harassment"] + df["sexual"] + df["sex"] + df["respect"] + df["threat"] + \
                          df["threatened"] + df["threatening"] + df["belittling"] + df["lesbian"] + df["transgender"] + \
                          df["depression"] + df["power imbalance"] + df["derogatory"] + df["gestures"] + df["offensive"]
df.drop(["safe", " dont feel safe ", " don't feel safe ", " do not feel safe ", " not safe ", "discrimination",
         "discriminatory language", "bad comments", "insult", "uncomfortable", "LGBTQ+", "gay", "psychological safety",
         "bullying", "harassment", "sexual", "sex", "respect", "threat", "threatened", "threatening", "belittling",
         "lesbian", "transgender", "depression", "power imbalance", "derogatory", "gestures", "offensive"],
        axis=1, inplace=True)

print("Done")

print(df.columns)

# Physical Safety

print("Categorizing Comments based on Physical Health Concerns\nPlease wait...")

phy_list = ["violence", "assault", "injury", "harm", "hurt", "physical"]

for physical in phy_list:
    df[physical] = df.astype(str).sum(axis=1).str.contains(physical)

df["Physical H.C"] = df["violence"] + df["assault"] + df["injury"] + df["harm"] + df["hurt"] + df["physical"]

df.drop(["violence", "assault", "injury", "harm", "hurt", "physical"], axis=1, inplace=True)

print("Done")

# Converting boolean to string
booleanDictionary = {True: 'TRUE', False: 'FALSE'}
df = df.replace(booleanDictionary)

# Deleting rows which do not belong to any categories

uncategorized_comments = df[
    (df["Accommodation"] == "FALSE") & (df["Transport"] == "FALSE") & (df["Cleanliness"] == "FALSE") & (
            df["Facilities"] == "FALSE") & (df["Food"] == "FALSE") & (df["Technology"] == "FALSE") & (
            df["Career Development"] == "FALSE") & (df["Safe Production System"] == "FALSE") & (
            df["Racism"] == "FALSE") & (df["Culture and Support"] == "FALSE") & (
            df["Indigenous"] == "FALSE") & (df["Health Concerns"] == "FALSE") & (
            df["Mandate"] == "FALSE") & (df["General"] == "FALSE") & (df["Utilities"] == "FALSE") & (
            df["Psychological H.C"] == "FALSE")
    ].index
df.drop(uncategorized_comments, inplace=True)

# Converting Answer column to string to avoid float encoding errors
df = df.astype({"Answer": str})

df.to_csv('Final_Sentiment_Analysis_1.csv')

# ---------VADER FOR SENTIMENT ANALYSIS----------------------------------------------------------------------------------

print("Initializing Sentiment Intensity Analyzer")

# Initialization
analyser = SentimentIntensityAnalyzer()

print("Calculating polarity scores")

# Creating new column for calculating polarity scores for pos,neg,neu and compound
df['Polarity Scores'] = df['Answer'].map(lambda x: analyser.polarity_scores(x))
print(df.head(10))

print("Assigning sentiments based on polarity scores")


# This function will calculate the sentiments
# def calculate(data):
#     if data['compound'] > 0.52:
#         if data['neg'] > 0:
#             if data['pos'] > data['neg']:
#                 data = 'pos'
#             else:
#                 data = 'neg'
#         else:
#             data = 'pos'
#
#     elif data['compound'] < 0.48:
#         data = 'neg'
#
#     elif 0.47 < data['compound'] < 0.53:
#         if data['neg'] > 0:
#             data = 'neg'
#         else:
#             data = 'neu'
#     return data

def calculate(data):
    if data['compound'] > 0.52:
        if data['neg'] > 0:
            data = 'neg'
        else:
            data = 'pos'

    elif data['compound'] < 0.48:
        data = 'neg'

    elif 0.47 < data['compound'] < 0.53:
        if data['neg'] > 0:
            data = 'neg'
        else:
            data = 'neu'
    return data


# Function call
df['New_Sentiment'] = df['Polarity Scores'].apply(calculate)
print(df)

# Save the file
df.to_csv('Sentiment_phy_vs_psy.csv')

# #Preprocessing the text file
#
# df_1 = df["Answer","New_Sentiment"]
# # df_1.to_csv('Sentiment_demo.txt', header=None, index=None, sep='\t', mode='a')

"""
From the past implemented script:

1. Number of Comments for SPS: 76
    pos: 26
    neg: 50

2. % of SPS comments based on overall:
 1.19%

**From this script***   
3. Neg Physical count: 220/25699 = 0.85%

4. Neg psy count:  3784/25699 =  14.72% w/o neu
                  + 55 (neu)
"""
