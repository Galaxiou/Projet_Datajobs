# Importation des librairies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px

# Importation des jeux de données
df_2020 = pd.read_csv('kaggle_survey_2020_responses.csv', low_memory= False)
df_2021 = pd.read_csv('kaggle_survey_2021_responses.csv', low_memory= False)

# Création des dictionnaires pour transformer df_2020 au format de df_2021 en vue de la fusion

# Création du dictionnaire d'équivalence pour transformer les Qx de 2020 vers ceux de 2021
dictQ = {'Q10_Part_13' : 'Q10_Part_16',
'Q12_Part_3' : 'Q12_Part_5',
'Q16_Part_15' : 'Q16_Part_17',
'Q20' : 'Q21',
'Q21' : 'Q22',
'Q22' : 'Q23',
'Q23_OTHER' : 'Q24_OTHER',
'Q23_Part_1' : 'Q24_Part_1',
'Q23_Part_2' : 'Q24_Part_2',
'Q23_Part_3' : 'Q24_Part_3',
'Q23_Part_4' : 'Q24_Part_4',
'Q23_Part_5' : 'Q24_Part_5',
'Q23_Part_6' : 'Q24_Part_6',
'Q23_Part_7' : 'Q24_Part_7',
'Q24' : 'Q25',
'Q25' : 'Q26',
'Q26_A_OTHER' : 'Q27_A_OTHER',
'Q26_A_Part_1' : 'Q27_A_Part_1',
'Q26_A_Part_10' : 'Q27_A_Part_10',
'Q26_A_Part_11' : 'Q27_A_Part_11',
'Q26_A_Part_2' : 'Q27_A_Part_2',
'Q26_A_Part_3' : 'Q27_A_Part_3',
'Q26_A_Part_4' : 'Q27_A_Part_4',
'Q26_A_Part_5' : 'Q27_A_Part_5',
'Q26_A_Part_6' : 'Q27_A_Part_6',
'Q26_A_Part_7' : 'Q27_A_Part_7',
'Q26_A_Part_8' : 'Q27_A_Part_8',
'Q26_A_Part_9' : 'Q27_A_Part_9',
'Q27_A_OTHER' : 'Q29_A_OTHER',
'Q27_A_Part_1' : 'Q29_A_Part_1',
'Q27_A_Part_7' : 'Q29_A_Part_3',
'Q27_A_Part_11' : 'Q29_A_Part_4',
'Q28_A_OTHER' : 'Q31_A_OTHER',
'Q28_A_Part_1' : 'Q31_A_Part_1',
'Q28_A_Part_4' : 'Q31_A_Part_2',
'Q28_A_Part_10' : 'Q31_A_Part_9',
'Q29_A_OTHER' : 'Q32_A_OTHER',
'Q29_A_Part_1' : 'Q32_A_Part_1',
'Q29_A_Part_11' : 'Q32_A_Part_11',
'Q29_A_Part_13' : 'Q32_A_Part_14',
'Q29_A_Part_14' : 'Q32_A_Part_15',
'Q29_A_Part_15' : 'Q32_A_Part_16',
'Q29_A_Part_16' : 'Q32_A_Part_17',
'Q29_A_Part_2' : 'Q32_A_Part_2',
'Q29_A_Part_17' : 'Q32_A_Part_20',
'Q29_A_Part_3' : 'Q32_A_Part_3',
'Q29_A_Part_4' : 'Q32_A_Part_4',
'Q29_A_Part_5' : 'Q32_A_Part_5',
'Q29_A_Part_6' : 'Q32_A_Part_6',
'Q29_A_Part_7' : 'Q32_A_Part_7',
'Q29_A_Part_8' : 'Q32_A_Part_8',
'Q30' : 'Q33',
'Q31_A_OTHER' : 'Q34_A_OTHER',
'Q31_A_Part_1' : 'Q34_A_Part_1',
'Q31_A_Part_10' : 'Q34_A_Part_10',
'Q31_A_Part_11' : 'Q34_A_Part_11',
'Q31_A_Part_12' : 'Q34_A_Part_12',
'Q31_A_Part_13' : 'Q34_A_Part_13',
'Q31_A_Part_14' : 'Q34_A_Part_16',
'Q31_A_Part_2' : 'Q34_A_Part_2',
'Q31_A_Part_3' : 'Q34_A_Part_3',
'Q31_A_Part_4' : 'Q34_A_Part_4',
'Q31_A_Part_5' : 'Q34_A_Part_5',
'Q31_A_Part_6' : 'Q34_A_Part_6',
'Q31_A_Part_7' : 'Q34_A_Part_7',
'Q31_A_Part_8' : 'Q34_A_Part_8',
'Q31_A_Part_9' : 'Q34_A_Part_9',
'Q32' : 'Q35',
'Q33_A_OTHER' : 'Q36_A_OTHER',
'Q33_A_Part_1' : 'Q36_A_Part_1',
'Q33_A_Part_2' : 'Q36_A_Part_2',
'Q33_A_Part_3' : 'Q36_A_Part_3',
'Q33_A_Part_4' : 'Q36_A_Part_4',
'Q33_A_Part_5' : 'Q36_A_Part_5',
'Q33_A_Part_6' : 'Q36_A_Part_6',
'Q33_A_Part_7' : 'Q36_A_Part_7',
'Q34_A_OTHER' : 'Q37_A_OTHER',
'Q34_A_Part_1' : 'Q37_A_Part_1',
'Q34_A_Part_2' : 'Q37_A_Part_2',
'Q34_A_Part_3' : 'Q37_A_Part_3',
'Q34_A_Part_4' : 'Q37_A_Part_4',
'Q34_A_Part_11' : 'Q37_A_Part_7',
'Q35_A_OTHER' : 'Q38_A_OTHER',
'Q35_A_Part_1' : 'Q38_A_Part_1',
'Q35_A_Part_10' : 'Q38_A_Part_11',
'Q35_A_Part_2' : 'Q38_A_Part_2',
'Q35_A_Part_3' : 'Q38_A_Part_3',
'Q35_A_Part_4' : 'Q38_A_Part_4',
'Q35_A_Part_5' : 'Q38_A_Part_5',
'Q35_A_Part_6' : 'Q38_A_Part_6',
'Q35_A_Part_7' : 'Q38_A_Part_7',
'Q35_A_Part_8' : 'Q38_A_Part_8',
'Q35_A_Part_9' : 'Q38_A_Part_9',
'Q36_OTHER' : 'Q39_OTHER',
'Q36_Part_1' : 'Q39_Part_1',
'Q36_Part_2' : 'Q39_Part_2',
'Q36_Part_3' : 'Q39_Part_3',
'Q36_Part_4' : 'Q39_Part_4',
'Q36_Part_5' : 'Q39_Part_5',
'Q36_Part_6' : 'Q39_Part_6',
'Q36_Part_7' : 'Q39_Part_7',
'Q36_Part_8' : 'Q39_Part_8',
'Q36_Part_9' : 'Q39_Part_9',
'Q37_OTHER' : 'Q40_OTHER',
'Q37_Part_1' : 'Q40_Part_1',
'Q37_Part_10' : 'Q40_Part_10',
'Q37_Part_11' : 'Q40_Part_11',
'Q37_Part_2' : 'Q40_Part_2',
'Q37_Part_3' : 'Q40_Part_3',
'Q37_Part_4' : 'Q40_Part_4',
'Q37_Part_5' : 'Q40_Part_5',
'Q37_Part_6' : 'Q40_Part_6',
'Q37_Part_7' : 'Q40_Part_7',
'Q37_Part_8' : 'Q40_Part_8',
'Q37_Part_9' : 'Q40_Part_9',
'Q38' : 'Q41',
'Q39_OTHER' : 'Q42_OTHER',
'Q39_Part_1' : 'Q42_Part_1',
'Q39_Part_10' : 'Q42_Part_10',
'Q39_Part_11' : 'Q42_Part_11',
'Q39_Part_2' : 'Q42_Part_2',
'Q39_Part_3' : 'Q42_Part_3',
'Q39_Part_4' : 'Q42_Part_4',
'Q39_Part_5' : 'Q42_Part_5',
'Q39_Part_6' : 'Q42_Part_6',
'Q39_Part_7' : 'Q42_Part_7',
'Q39_Part_8' : 'Q42_Part_8',
'Q39_Part_9' : 'Q42_Part_9',
'Q9_Part_11' : 'Q9_Part_12',
}

# Colonnes à ajouter en full NAN dans le sondage de 2020
Ncol2020 = ['Q10_Part_13',
'Q10_Part_14',
'Q10_Part_15',
'Q12_Part_3',
'Q12_Part_4',
'Q16_Part_15',
'Q16_Part_16',
'Q20',
'Q28',
'Q29_A_Part_2',
'Q30_A_OTHER',
'Q30_A_Part_1',
'Q30_A_Part_2',
'Q30_A_Part_3',
'Q30_A_Part_4',
'Q30_A_Part_5',
'Q30_A_Part_6',
'Q30_A_Part_7',
'Q31_A_Part_3',
'Q31_A_Part_4',
'Q31_A_Part_5',
'Q31_A_Part_6',
'Q31_A_Part_7',
'Q31_A_Part_8',
'Q32_A_Part_10',
'Q32_A_Part_12',
'Q32_A_Part_13',
'Q32_A_Part_18',
'Q32_A_Part_19',
'Q32_A_Part_9',
'Q34_A_Part_14',
'Q34_A_Part_15',
'Q37_A_Part_5',
'Q37_A_Part_6',
'Q38_A_Part_10',
'Q9_Part_11',

]

# Colonnes "B" à supprimer de 2021
Scol2021 = ['Q27_B_OTHER',
'Q27_B_Part_1',
'Q27_B_Part_10',
'Q27_B_Part_11',
'Q27_B_Part_2',
'Q27_B_Part_3',
'Q27_B_Part_4',
'Q27_B_Part_5',
'Q27_B_Part_6',
'Q27_B_Part_7',
'Q27_B_Part_8',
'Q27_B_Part_9',
'Q29_B_OTHER',
'Q29_B_Part_1',
'Q29_B_Part_2',
'Q29_B_Part_3',
'Q29_B_Part_4',
'Q30_B_OTHER',
'Q30_B_Part_1',
'Q30_B_Part_2',
'Q30_B_Part_3',
'Q30_B_Part_4',
'Q30_B_Part_5',
'Q30_B_Part_6',
'Q30_B_Part_7',
'Q31_B_OTHER',
'Q31_B_Part_1',
'Q31_B_Part_2',
'Q31_B_Part_3',
'Q31_B_Part_4',
'Q31_B_Part_5',
'Q31_B_Part_6',
'Q31_B_Part_7',
'Q31_B_Part_8',
'Q31_B_Part_9',
'Q32_B_OTHER',
'Q32_B_Part_1',
'Q32_B_Part_10',
'Q32_B_Part_11',
'Q32_B_Part_12',
'Q32_B_Part_13',
'Q32_B_Part_14',
'Q32_B_Part_15',
'Q32_B_Part_16',
'Q32_B_Part_17',
'Q32_B_Part_18',
'Q32_B_Part_19',
'Q32_B_Part_2',
'Q32_B_Part_20',
'Q32_B_Part_3',
'Q32_B_Part_4',
'Q32_B_Part_5',
'Q32_B_Part_6',
'Q32_B_Part_7',
'Q32_B_Part_8',
'Q32_B_Part_9',
'Q34_B_OTHER',
'Q34_B_Part_1',
'Q34_B_Part_10',
'Q34_B_Part_11',
'Q34_B_Part_12',
'Q34_B_Part_13',
'Q34_B_Part_14',
'Q34_B_Part_15',
'Q34_B_Part_16',
'Q34_B_Part_2',
'Q34_B_Part_3',
'Q34_B_Part_4',
'Q34_B_Part_5',
'Q34_B_Part_6',
'Q34_B_Part_7',
'Q34_B_Part_8',
'Q34_B_Part_9',
'Q36_B_OTHER',
'Q36_B_Part_1',
'Q36_B_Part_2',
'Q36_B_Part_3',
'Q36_B_Part_4',
'Q36_B_Part_5',
'Q36_B_Part_6',
'Q36_B_Part_7',
'Q37_B_OTHER',
'Q37_B_Part_1',
'Q37_B_Part_2',
'Q37_B_Part_3',
'Q37_B_Part_4',
'Q37_B_Part_5',
'Q37_B_Part_6',
'Q37_B_Part_7',
'Q38_B_OTHER',
'Q38_B_Part_1',
'Q38_B_Part_10',
'Q38_B_Part_11',
'Q38_B_Part_2',
'Q38_B_Part_3',
'Q38_B_Part_4',
'Q38_B_Part_5',
'Q38_B_Part_6',
'Q38_B_Part_7',
'Q38_B_Part_8',
'Q38_B_Part_9',
]

# Colonnes à supprimer de 2020 pour le ramener à la taille de 2021
Scol2020 = ['Q27_A_Part_2',
'Q27_A_Part_4',
'Q27_A_Part_3',
'Q27_A_Part_5',
'Q27_A_Part_6',
'Q27_A_Part_8',
'Q27_A_Part_9',
'Q27_A_Part_10',
'Q27_B_Part_2',
'Q27_B_Part_4',
'Q27_B_Part_3',
'Q27_B_Part_5',
'Q27_B_Part_6',
'Q27_B_Part_8',
'Q27_B_Part_9',
'Q27_B_Part_10',
'Q28_A_Part_2',
'Q28_A_Part_3',
'Q28_A_Part_5',
'Q28_A_Part_6',
'Q28_A_Part_7',
'Q28_A_Part_8',
'Q28_A_Part_9',
'Q28_B_Part_2',
'Q28_B_Part_3',
'Q28_B_Part_5',
'Q28_B_Part_6',
'Q28_B_Part_7',
'Q28_B_Part_8',
'Q28_B_Part_9',
'Q29_A_Part_9',
'Q29_A_Part_10',
'Q29_A_Part_12',
'Q29_B_Part_9',
'Q29_B_Part_10',
'Q29_B_Part_12',
'Q34_A_Part_5',
'Q34_A_Part_6',
'Q34_A_Part_7',
'Q34_A_Part_8',
'Q34_A_Part_9',
'Q34_A_Part_10',
'Q34_B_Part_5',
'Q34_B_Part_6',
'Q34_B_Part_7',
'Q34_B_Part_8',
'Q34_B_Part_9',
'Q34_B_Part_10',
'Q26_B_OTHER',
'Q26_B_Part_1',
'Q26_B_Part_10',
'Q26_B_Part_11',
'Q26_B_Part_2',
'Q26_B_Part_3',
'Q26_B_Part_4',
'Q26_B_Part_5',
'Q26_B_Part_6',
'Q26_B_Part_7',
'Q26_B_Part_8',
'Q26_B_Part_9',
'Q27_B_OTHER',
'Q27_B_Part_1',
'Q27_B_Part_7',
'Q27_B_Part_11',
'Q28_B_OTHER',
'Q28_B_Part_1',
'Q28_B_Part_4',
'Q28_B_Part_10',
'Q29_B_OTHER',
'Q29_B_Part_1',
'Q29_B_Part_11',
'Q29_B_Part_13',
'Q29_B_Part_14',
'Q29_B_Part_15',
'Q29_B_Part_16',
'Q29_B_Part_2',
'Q29_B_Part_17',
'Q29_B_Part_3',
'Q29_B_Part_4',
'Q29_B_Part_5',
'Q29_B_Part_6',
'Q29_B_Part_7',
'Q29_B_Part_8',
'Q31_B_OTHER',
'Q31_B_Part_1',
'Q31_B_Part_10',
'Q31_B_Part_11',
'Q31_B_Part_12',
'Q31_B_Part_13',
'Q31_B_Part_14',
'Q31_B_Part_2',
'Q31_B_Part_3',
'Q31_B_Part_4',
'Q31_B_Part_5',
'Q31_B_Part_6',
'Q31_B_Part_7',
'Q31_B_Part_8',
'Q31_B_Part_9',
'Q33_B_OTHER',
'Q33_B_Part_1',
'Q33_B_Part_2',
'Q33_B_Part_3',
'Q33_B_Part_4',
'Q33_B_Part_5',
'Q33_B_Part_6',
'Q33_B_Part_7',
'Q34_B_OTHER',
'Q34_B_Part_1',
'Q34_B_Part_2',
'Q34_B_Part_3',
'Q34_B_Part_4',
'Q34_B_Part_11',
'Q35_B_OTHER',
'Q35_B_Part_1',
'Q35_B_Part_10',
'Q35_B_Part_2',
'Q35_B_Part_3',
'Q35_B_Part_4',
'Q35_B_Part_5',
'Q35_B_Part_6',
'Q35_B_Part_7',
'Q35_B_Part_8',
'Q35_B_Part_9',
]

# Préparation des données

# Suppression des colonnes en trop de 2020
new_df_2020 = df_2020.drop(Scol2020, axis = 1)

# Suppression des colonnes en trop de 2021
new_df_2021 = df_2021.drop(Scol2021, axis = 1)

# Remplacement des Qx 2020 par Qx2021
new_df_2020 = new_df_2020.rename(columns = dictQ)

# Ajout des nouvelles colonnes
new_df_2020[Ncol2020] = np.NAN

# Suppression de la ligne des questions pour 2020
new_df_2020 = new_df_2020.drop(0)

# Ajout de la colonne "Year" pour le sondage de 2020
new_df_2020['Year'] = 2020

# Ajout de la colonne "Year" pour le sondage de 2021
new_df_2021['Year'] = 2021

# Fusion des deux sondages
full_df = pd.concat([new_df_2021, new_df_2020])

# Suppression des lignes qui n'ont pas 10 réponses ou plus
full_df = full_df.dropna(thresh=(10))

# Conversion de la colonne durée du sondage, puis suppression des lignes qui ont pris strictement moins de 120s pour répondre au questionnaire
full_df = full_df.replace(['Duration (in seconds)'],[1000])
full_df['Time from Start to Finish (seconds)'] = full_df['Time from Start to Finish (seconds)'].astype(int)
full_df = full_df[full_df['Time from Start to Finish (seconds)'] > 120]

# Suppression de la colonne durée du sondage
full_df = full_df.drop('Time from Start to Finish (seconds)', axis = 1)

# Réaffectation des index pour eviter les doublons
full_df = full_df.reset_index().drop('index', axis = 1)


# Data cleaning

# Correction des valeurs dans la colonne Q4 : 'Professional degree' => 'Professional doctorate'
full_df['Q4'] = full_df['Q4'].replace(['Professional degree'], ['Professional doctorate'])

# Correction des valeurs dans la colonne Q5 : 'Product/Project Manager' => 'Product Manager'
full_df['Q5'] = full_df['Q5'].replace(['Product/Project Manager'], ['Product Manager'])

# Correction des valeurs dans la colonne Q6 : '1-3 years' => '1-2 years'
full_df['Q6'] = full_df['Q6'].replace(['1-3 years'], ['1-2 years'])

# Correction des valeurs dans la colonne Q9_Part_3 : ' Visual Studio ' => 'Visual Studio'
full_df['Q9_Part_3'] = full_df['Q9_Part_3'].replace([' Visual Studio '], ['Visual Studio'])

# Correction des valeurs dans la colonne Q9_Part_4 : ' Visual Studio Code (VSCode) ' => 'Visual Studio Code (VSCode)'
full_df['Q9_Part_4'] = full_df['Q9_Part_4'].replace([' Visual Studio Code (VSCode) '], ['Visual Studio Code (VSCode)'])

# Correction des valeurs dans la colonne Q10_Part_8 : ' Amazon Sagemaker Studio ' => ' Amazon Sagemaker Studio Notebooks '
full_df['Q10_Part_8'] = full_df['Q10_Part_8'].replace([' Amazon Sagemaker Studio '], [' Amazon Sagemaker Studio Notebooks '])

# Correction des valeurs dans la colonne Q10_Part_10 : 'Google Cloud AI Platform Notebooks ' => 'Google Cloud Notebooks (AI Platform / Vertex AI) '
full_df['Q10_Part_10'] = full_df['Q10_Part_10'].replace(['Google Cloud AI Platform Notebooks '], 
                                                        ['Google Cloud Notebooks (AI Platform / Vertex AI) '])

# Correction des valeurs dans la colonne Q10_Part_11 : 'Google Cloud Datalab' => 'Google Cloud Datalab Notebooks'
full_df['Q10_Part_11'] = full_df['Q10_Part_11'].replace(['Google Cloud Datalab'], ['Google Cloud Datalab Notebooks'])

# Correction des valeurs dans la colonne Q11
full_df['Q11'] = full_df['Q11'].replace(['A laptop'], ['A personal computer or laptop'])
full_df['Q11'] = full_df['Q11'].replace(['A personal computer / desktop'], ['A personal computer or laptop'])

# Correction des valeurs dans la colonne Q12_Part_1 : ' NVIDIA GPUs ' => 'GPUs'
full_df['Q12_Part_1'] = full_df['Q12_Part_1'].replace([' NVIDIA GPUs '], ['GPUs'])

# Correction des valeurs dans la colonne Q12_Part_2 : ' Google Cloud TPUs ' => 'TPUs'
full_df['Q12_Part_2'] = full_df['Q12_Part_2'].replace([' Google Cloud TPUs '], ['TPUs'])

# Correction des valeurs dans la colonne Q25
full_df['Q25'] = full_df['Q25'].replace(['$0-999'], ['0-999'])
full_df['Q25'] = full_df['Q25'].replace(['>$1,000,000'], ['>1,000,000'])
full_df['Q25'] = full_df['Q25'].replace(['> $500,000'], ['> 500,000'])
full_df['Q25'] = full_df['Q25'].replace(['$500,000-999,999'], ['500,000-999,999'])
full_df['Q25'] = full_df['Q25'].replace(['300,000-499,999'], ['300,000-500,000'])
full_df['Q25'] = full_df['Q25'].replace(['500,000-999,999'], ['> 500,000'])
full_df['Q25'] = full_df['Q25'].replace(['>1,000,000'], ['> 500,000'])

# Correction des valeurs dans la colonne Q29_A_Part_1 : ' Amazon EC2 ' => ' Amazon Elastic Compute Cloud (EC2) '
full_df['Q29_A_Part_1'] = full_df['Q29_A_Part_1'].replace([' Amazon EC2 '], [' Amazon Elastic Compute Cloud (EC2) '])

# Correction des valeurs dans la colonne Q32_A_Part_2 : 'PostgresSQL ' => 'PostgreSQL '
full_df['Q32_A_Part_2'] = full_df['Q32_A_Part_2'].replace(['PostgresSQL '], ['PostgreSQL '])

# Correction des valeurs dans la colonne Q33 : 'PostgresSQL ' => 'PostgreSQL '
full_df['Q33'] = full_df['Q33'].replace(['PostgresSQL '], ['PostgreSQL '])

# Correction des valeurs dans la colonne Q34_A_Part_7 : 'Einstein Analytics' => 'Tableau CRM'
full_df['Q34_A_Part_7'] = full_df['Q34_A_Part_7'].replace(['Einstein Analytics'], ['Tableau CRM'])

# Correction des valeurs dans la colonne Q35 : 'Einstein Analytics' => 'Tableau CRM'
full_df['Q35'] = full_df['Q35'].replace(['Einstein Analytics'], ['Tableau CRM'])

# Correction des valeurs dans la colonne Q36_A_Part_6 : 'Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)' =>
#                                                       'Automation of full ML pipelines (e.g. Google AutoML, H2O Driverless AI)'
full_df['Q36_A_Part_6'] = full_df['Q36_A_Part_6'].replace(['Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)'],
                                                          ['Automation of full ML pipelines (e.g. Google AutoML, H2O Driverless AI)'])

# Correction des valeurs dans la colonne Q37_A_Part_2 : ' H20 Driverless AI' => ' H2O Driverless AI'
full_df['Q37_A_Part_2'] = full_df['Q37_A_Part_2'].replace([' H20 Driverless AI  '], [' H2O Driverless AI  '])

# Correction des valeurs dans la colonne Q38_A_Part_8 : ' Trains ' => ' ClearML '
full_df['Q38_A_Part_8'] = full_df['Q38_A_Part_8'].replace([' Trains '], [' ClearML '])


# Préparation des données pour la data viz


# Création d'un dataframe listQ contenant les Qx et les intitulés de chaque question

# Récupération de tous les Qx et intitulés depuis le dataframe global
listQ = full_df.iloc[0,:-1].reset_index().rename({'index' : 'Qx', 0 : 'Question'}, axis = 1)
# Suppression des Qx en double (exemple : s'il y a 'Q1_Part1' et 'Q1_Part2', seul 'Q1' sera gardé)
listQ['Qx'] = listQ['Qx'].apply(lambda i: i.split('_')[0]).drop_duplicates()
# Suppression de la partie "réponse" de chaque intitulé de question ainsi que du "select all that apply" situé après un "?"
listQ['Question'] = listQ['Question'].apply(lambda i: i.split(':')[0].split('?')[0])
# Suppression des NAN
listQ = listQ.dropna().set_index('Qx')


# Création de dataframe filtrés par type d'emploi
filt_df_DE = full_df.loc[(full_df['Q5'] == 'Data Engineer')]
filt_df_BA = full_df.loc[(full_df['Q5'] == 'Business Analyst')]
filt_df_MLE = full_df.loc[(full_df['Q5'] == 'Machine Learning Engineer')]
filt_df_RS = full_df.loc[(full_df['Q5'] == 'Research Scientist')]
filt_df_DA = full_df.loc[(full_df['Q5'] == 'Data Analyst')]
filt_df_SE = full_df.loc[(full_df['Q5'] == 'Software Engineer')]
filt_df_DS = full_df.loc[(full_df['Q5'] == 'Data Scientist')]


# Création d'un dataframe df_viz contenant la somme de chaque réponse par métier

# On retire temporairement la colonne des jobs (Q5) et on retire celle de l'année
df_viz = full_df.drop(['Q5', 'Year'], axis = 1)

# Nous identifions les colonnes multi (avec plusieurs réponses) et les colonnes mono (avec une seule réponse)
multi_Qx = df_viz.columns[df_viz.nunique() > 2]
mono_Qx = df_viz.columns[df_viz.nunique() <= 2]

# Transformation en true et false des valeurs des colonnes mono
df_viz.loc[1:,mono_Qx] = df_viz.loc[1:,mono_Qx].notnull()

# Changement des noms des colonnes mono pour inclure la réponse
for i in df_viz[mono_Qx].columns:
    df_viz = df_viz.rename({i : i.split('_')[0] + '_' + df_viz[i][0].split('-')[2]}, axis = 1)

# Retrait de la ligne contenant l'intitulé des questions
df_viz = df_viz.drop(0)

# Get dummies des colonnes à réponse multiple pour les convertir en 0 et 1 que l'on pourra compter
df_viz_multi = pd.get_dummies(df_viz[multi_Qx])

# Fusion des données de toutes les questions et transformation en 0 et 1 avec un astype
df_viz = df_viz.drop(multi_Qx, axis = 1).join(df_viz_multi).astype('int')

# On réintègre les colonnes des jobs et de l'année
df_viz = pd.concat([full_df['Q5'].drop(0), df_viz], axis = 1)

# Création d'un dataframe contenant les sommes de chaque réponse pour les 7 métiers les plus représentés ainsi que pour le total
df_viz = df_viz.groupby('Q5').sum().astype('int').join(df_viz['Q5'].value_counts()).rename({'Q5' : 'Q5_count'}, axis = 1)
df_viz = df_viz.append(df_viz.sum().rename('Total'))
df_viz = df_viz.drop(['Student', 'Other', 'Currently not employed'])
df_viz = df_viz.sort_values(by = 'Q5_count', ascending = False).head(8)



# Fonctions pour la data viz


# Fonctions de mise en forme / filtrage des données

def QxToQuestion(Qx):
    """
    Cette fonction sert à récupérer l'intitulé d'une question à partir de son ID (au format 'Qx').

    Input :
        Qx (string) : par exemple 'Q1' ou 'Q36'
        
    Output :
        L'intitulé de la question correspondante au format string.
    """
    return listQ.loc[Qx,'Question']



def MultiQCount(question, filt_df):
    """
    Fonction permettant de compter les différentes réponses à une question, ceci même lorsque ses réponses sont divisées en plusieurs colonnes.
    Attention, toutes les colonnes doivent se suivre et avoir le même enoncé sinon le code ne fonctionne pas.
    
    Input :
        'question' (string) : l'intitulé de la question.
        'filt_df' (dataframe) : Le dataframe utilisé. Celui-ci peut-être filtré en fonction d'un critère particulier
            Exemple: filt_df = full_df.loc[full_df['Q5'] == 'Data Scientist']

    Output :
        Cette fonction renvoie un dataframe avec deux colonnes :
            'Answer' contenant les différentes réponses associées à 'question'
            'Count' contenant le nombre de fois que cette réponse apparait dans 'filt_df'
        Le dataframe a autant de lignes qu'il y a de réponses à la question.
    """
    #Listing de l'emplacement des colonnes concernées
    QColList = []
    for i, e in enumerate(full_df.iloc[0,:-1]):
        if question in e:
            QColList.append(i)

    #Début et fin de la zone à selectionner dans le DF
    endQ = max(QColList) + 1
    startQ = min(QColList)

    #Pour les question à une colonne, un head(-1) a été rajouté afain de retirer l'intitulé de la question du décompte
    if startQ == endQ -1:
        QuestCount = full_df.iloc[0:, startQ].value_counts().reset_index().rename({'index' : 'Answer', full_df.columns[startQ] :'Count'}, axis = 1).head(-1)
    else:
        QuestCount = full_df.iloc[0, startQ : endQ].apply(lambda x: x.split('-')[2]).to_frame().rename({0:'Answer'}, axis = 1).join(
        filt_df.iloc[1:, startQ : endQ].notna().sum().to_frame().rename({0:'Count'}, axis = 1)).sort_values(by = 'Count', ascending = False)    

    return(QuestCount)



def MultiQCountPerc(question, filt_df):
    """
    Fonction permettant de calculer la proportion de chaque réponse à une question, ceci même lorsque ses réponses sont divisées en plusieurs colonnes.
    Attention, toutes les colonnes doivent se suivre et avoir le même enoncé sinon le code ne fonctionne pas.
    
    Input :
        'question' (string) : L'intitulé de la question
        'filt_df' (dataframe) : Le dataframe utilisé. Celui-ci peut-être filtré en fonction d'un critère particulier
            Exemple: filt_df = full_df.loc[full_df['Q5'] == 'Data Scientist']

    Output :
        Cette fonction renvoie un dataframe avec deux colonnes :
            'Answer' contenant les différentes réponses associées à 'question'
            'Count' contenant le nombre de fois que cette réponse apparait dans 'filt_df'
        Le dataframe a autant de lignes qu'il y a de réponses à la question.
    """
    #Listing de l'emplacement des colonnes concernées
    QColList = []
    for i, e in enumerate(full_df.iloc[0,:-1]):
        if question in e:
            QColList.append(i)

    #Début et fin de la zone à selectionner dans le DF
    endQ = max(QColList) + 1
    startQ = min(QColList)

    #Pour les question à une colonne, un head(-1) a été rajouté afain de retirer l'intitulé de la question du décompte
    if startQ == endQ -1:
        QuestCount = full_df.iloc[:, startQ].value_counts().reset_index().rename({'index' : 'Answer', filt_df.columns[startQ] : 'Count'}, axis = 1).head(-1)
        QuestCount['Count'] = round(QuestCount['Count'] / filt_df.shape[0]*100 , 2)
    else:
        QuestCount = full_df.iloc[0, startQ : endQ].apply(lambda x: x.split('-')[2]).to_frame().rename({0 : 'Answer'}, axis = 1).join(
        filt_df.iloc[1:, startQ : endQ].notna().sum().to_frame().rename({0 : 'Count'}, axis = 1)).sort_values(by = 'Count', ascending = False)    
        QuestCount['Count'] = round(QuestCount['Count'] / filt_df.shape[0]*100 , 2)

    return(QuestCount)



def NewBarhCountList(question, filt_df):
    """
    Dans le cas des questions présentées sur plusieurs colonnes,
    cette fonction retourne la proportion en pourcentage de chacune des réponses sous forme de liste.
    Cette donnée nous sera utile pour la génération d’une visualisation via la fonction survey qui sera présentée plus bas.
    Les paramètres de cette fonction sont l’intitulé de la question ainsi que le dataframe sélectionné.
    """
    QColList = []
    OtherQ = 0
    for i, e in enumerate(full_df.iloc[0,:-1]):
        if question in e:
            QColList.append(i)
            if "None" in e:
                OtherQ = i

    #Début et fin de la zone à selectionner dans le DF
    endQ = max(QColList) + 1
    startQ = min(QColList)

    ListResults = [0, 0, 0, 0, 0, 0]

    for i in range(len(filt_df)):
        if (filt_df.iloc[i,OtherQ] == "None"):
            count = 0
        else:
            count = filt_df.iloc[i, startQ : endQ].count()
        ListResults[-(count+1) if (count < 5) else -6] +=1

    ListResults = [round(i2/len(filt_df)*100,2) for i2 in ListResults]
    
    return ListResults



def NewBarhCountListReduced(question, filt_df):
    """
    Cette fonction est une version réduite de la fonction NewBarhCountList.
    Elle renvoie une liste de 3 pourcentages seulement (contre 6 pour la version normale de la fonction).
    Nous utilisons cette fonction spécifiquement pour afficher le nombre de réponse à la Q34 car elle ne comporte que peu de réponses intéressantes.
    """
    QColList = []
    OtherQ = 0
    for i, e in enumerate(full_df.iloc[0,:-1]): # ':-1' dans le iloc pour ignorer la dernière colonne : la colonne 'Year'
        if question in e:
            QColList.append(i)
            if "None" in e:
                OtherQ = i

    #Début et fin de la zone à selectionner dans le DF
    endQ = max(QColList) + 1
    startQ = min(QColList)

    ListResults = [0, 0, 0]

    for i in range(len(filt_df)):
        if (filt_df.iloc[i,OtherQ] == "None"):
            count = 0
        else:
            count = filt_df.iloc[i, startQ : endQ].count()
        ListResults[-(count+1) if (count < 2) else -3] +=1

    ListResults = [round(i2/len(filt_df)*100,2) for i2 in ListResults]
    
    return ListResults



# Fonctions permettant la génération des graphiques


def BarGraph(FQxToQuestion, filt_df):
    """
    Fonction retournant un histogramme horizontal affichant toutes les réponses différentes
    à une même question ainsi que le compte de chacune de ces réponses.
    Les paramètres de cette fonction sont l’intitulé de la question ainsi que le dataframe sélectionné.
    """
    plt.figure(figsize = (15, 7))
    plt.barh(MultiQCount(FQxToQuestion, filt_df)['Answer'], (MultiQCount(FQxToQuestion, filt_df)['Count']), color = '#b96a7c' )
    plt.title(FQxToQuestion)
    plt.xlabel('Count')
    return plt.show();



def BarGraphPerc(FQxToQuestion, filt_df):
    """
    Fonction retournant un histogramme horizontal affichant toutes les réponses différentes
    à une même question ainsi que le ratio en pourcentage de chacune de ces réponses.
    Les paramètres de cette fonction sont l’intitulé de la question ainsi que le dataframe sélectionné.
    """
    fig, ax = plt.subplots()
    ax.barh(MultiQCountPerc(FQxToQuestion, filt_df)['Answer'], (MultiQCountPerc(FQxToQuestion, filt_df)['Count']), color = '#b96a7c' )
    if filt_df['Q5'].nunique() < 2:
        title = filt_df['Q5'].unique()[0] + " - " + FQxToQuestion
    else:
        title = FQxToQuestion
    plt.title(title)
    plt.xlabel('Pourcentage')
    return st.pyplot(fig);



def BarYearGraphPerc(FQxToQuestion, filt_df):
    """
    Fonction retournant un histogramme horizontal affichant toutes les réponses différentes
    à une même question ainsi que le ratio en pourcentage de chacune de ces réponses avec dissociation des données de 2021 et 2020.
    Les paramètres de cette fonction sont l’intitulé de la question ainsi que le dataframe sélectionné.
    """
    df = pd.DataFrame(dict(graph = MultiQCountPerc(FQxToQuestion, filt_df)['Answer'], 
                        m = MultiQCountPerc(FQxToQuestion, filt_df.loc[(filt_df['Year'] == 2021)])['Count'], 
                        n = MultiQCountPerc(FQxToQuestion, filt_df.loc[(filt_df['Year'] == 2020)])['Count'])).sort_values('m', ascending = True)

    ind = np.arange(len(df))
    width = 0.4

    fig, ax = plt.subplots()
    ax.barh(ind, df["n"], width, color='#b4ceff', label='2020')
    ax.barh(ind + width, df["m"], width, color='#b96a7c', label='2021')
    ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
    ax.legend()
    ax.set_xlim(0, 105)
    fig.set_size_inches(17.5, 10)
    if filt_df['Q5'].nunique() < 2:
        title = filt_df['Q5'].unique()[0] + " - " + FQxToQuestion
    else:
        title = FQxToQuestion
    plt.title(title)
    plt.xlabel('Pourcentage')
    
    return st.pyplot(fig);



def survey(results, category_names):
    """
    Fonction retournant un histogramme horizontal indiquant la proportion de chacune des réponses à une même question en fonction du poste des répondants.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    base = plt.cm.get_cmap('Blues_r')
    category_colors = base(np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(12.5, 5))
    ax.invert_yaxis()
    ax.xaxis.set_ticks(np.arange(0, 101, 5))
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)
        
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'dimgray'
        ax.bar_label(rects, label_type='center', color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0.1, 1),
              loc='lower left', fontsize='small')                         

    return plt.show();



def TopAnswersPerJob(Qx, n_top = 3):
    """
    Cette fonction permet de générer un graphique représentant les proportions des n_top principales réponses à Qx pour chaque métier. Elle réalise dans l'ordre :
    - l'identification des n_top principales réponses de la question
    - le stockage dans un dataframe des proportions de ces réponses pour chaque métier
    - l'affichage du graphique correspondant
    Input :
        Qx (string) : L'ID de la question (ex : 'Q6', 'Q27', etc). Toutes les questions sont acceptées par la fonction de Q1 à Q42, sauf Q5.
        n_top (int) : correspond au nombre de réponses à la question que l'on souhaite afficher. Doit valoir 2, 3 ou 4 (par défaut 3).
    Output :
        Aucune, génère un graphe.
    """
    df = df_viz
    
    # On récupère les principales réponses de Qx
    counts = {}
    for i in df.columns:
        if Qx == i.split('_')[0]:
            counts[i] = df[i].sum()
    top1 = max(counts, key = counts.get)
    del counts[top1]
    top2 = max(counts, key = counts.get)
    del counts[top2]
    top3 = max(counts, key = counts.get)
    del counts[top3]
    top4 = max(counts, key = counts.get)
    
    # Détermine la proportion des 3 principales réponses pour chaque métier
    data = pd.DataFrame([df[top1], df[top2], df[top3], df[top4]])
    for i in data.columns:
        data[i] = data[i].apply(lambda x: round(x/df['Q5_count'][i],2)*100)
    data = data.T

    # Affichage du graphique
    ind = np.arange(len(data))
    width = 0.6/n_top
    fig, ax = plt.subplots()
    ax.barh(ind + (n_top-1)*width, data[top1], width, color = 'darkseagreen', label = top1.split('_')[1])
    ax.barh(ind + (n_top-2)*width, data[top2], width, color='cornflowerblue', label = top2.split('_')[1])
    if n_top >= 3:
        ax.barh(ind + (n_top-3)*width, data[top3], width, color='lightcoral', label = top3.split('_')[1])
    if n_top == 4:
        ax.barh(ind, data[top4], width, color='gold', label = top4.split('_')[1])
    ax.set(yticks = ind + width, yticklabels = data.index, ylim = [2*width - 1, len(data)])
    ax.legend(fontsize = 15)
    ax.set_xlim(0, 105)
    fig.set_size_inches(17.5, 10)
    plt.title(QxToQuestion(Qx), fontsize = 20)
    plt.xlabel('Pourcentage', fontsize = 18)
    ax.tick_params(labelsize= 15)

    return st.pyplot(fig);



def CountMap(df, title, valeur_max):
    """
    Fonction permettant de retourner une carte en couleur mettant en avant le pays des répondants.
    """
    fig = px.choropleth(df, 
                    locations = 'Answer',  
                    color = 'Count',
                    locationmode = 'country names', 
                    color_continuous_scale = 'Blues',
                    title = title,
                    range_color = [0, valeur_max])
    fig.update(layout=dict(title=dict(x=0.5)))
    fig.show()
    

import streamlit as st    

st.sidebar.title("Sommaire")
pages = ["Introduction","Dataviz","Modélisation"]
page = st.sidebar.radio("Aller vers", pages)

if page==pages[0]:
    st.title("Projet Datajobs")
    
    st.write("## Introduction au sujet")
    
    st.write("Voici un aperçu du Dataframe")
    
    st.dataframe(df_2020.head())
    
    if st.checkbox("Afficher les variables manquantes "):
        st.dataframe(df_2020.isna().sum())
        
elif page==pages[1]:
    st.write("## Dataviz")
    
    listjob = ('Program/Project Manager', 'Software Engineer',
       'Research Scientist', 'Currently not employed', 'Student',
       'Data Scientist', 'Data Analyst', 'Machine Learning Engineer',
       'Business Analyst', 'Data Engineer', 'Product Manager',
       'Statistician', 'Developer Relations/Advocacy',
       'DBA/Database Engineer')
    
    listQuestion = ("Q1  :  What is your age? (# years)",
                    "Q2  :  What is your gender",
                    "Q3  :  In which country do you currently reside?",
                    "Q4  :  What is the highest level of formal education that you have attained or plan to attain within the next 2 years?",
                    "Q5  :  Select the title most similar to your current role (or most recent title if retired)?",
                    "Q6  :  For how many years have you been writing code and/or programming?",
                    "Q7  :  What programming languages do you use on a regular basis?",
                    "Q8  :  What programming language would you recommend an aspiring data scientist to learn first?",
                    "Q9  :  Which of the following integrated development environments (IDE's) do you use on a regular basis?",
                    "Q10  :  Which of the following hosted notebook products do you use on a regular basis?",
                    "Q11  :  What type of computing platform do you use most often for your data science projects?",
                    "Q12  :  Which types of specialized hardware do you use on a regular basis?",
                    "Q13  :  Approximately how many times have you used a TPU (tensor processing unit)?",
                    "Q14  :  What data visualization libraries or tools do you use on a regular basis?",
                    "Q15  :  For how many years have you used machine learning methods?",
                    "Q16  :  Which of the following machine learning frameworks do you use on a regular basis?",
                    "Q17  :  Which of the following ML algorithms do you use on a regular basis?",
                    "Q18  :  Which categories of computer vision methods do you use on a regular basis?",
                    "Q19  :  Which of the following natural language processing (NLP) methods do you use on a regular basis?",
                    "Q20  :  In what industry is your current employer/contract (or your most recent employer if retired)?",
                    "Q21  :  What is the size of the company where you are employed?",
                    "Q22  :  Approximately how many individuals are responsible for data science workloads at your place of business?",
                    "Q23  :  Does your current employer incorporate machine learning methods into their business?",
                    "Q24  :  Select any activities that make up an important part of your role at work?",
                    "Q25  :  What is your current yearly compensation (approximate $USD)?",
                    "Q26  :  Approximately how much money have you (or your team) spent on machine learning and/or cloud computing services at home (or at work) in the past 5 years (approximate $USD)?",
                    "Q27  :  Which of the following cloud computing platforms do you use on a regular basis?",
                    "Q28  :  Of the cloud platforms that you are familiar with, which has the best developer experience (most enjoyable to use)?",
                    "Q29  :  Do you use any of the following cloud computing products on a regular basis?",
                    "Q30  :  Do you use any of the following data storage products on a regular basis?",
                    "Q31  :  Do you use any of the following managed machine learning products on a regular basis?",
                    "Q32  :  Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis?",
                    "Q33  :  Which of the following big data products (relational database, data warehouse, data lake, or similar) do you use most often?",
                    "Q34  :  Which of the following business intelligence tools do you use on a regular basis?",
                    "Q35  :  Which of the following business intelligence tools do you use most often?",
                    "Q36  :  Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?",
                    "Q37  :  Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?",
                    "Q38  :  Do you use any tools to help manage machine learning experiments?",
                    "Q39  :  Where do you publicly share your data analysis or machine learning applications?",
                    "Q40  :  On which platforms have you begun or completed data science courses?",
                    "Q41  :  What is the primary tool that you use at work or school to analyze data?",
                    "Q42  :  Who/what are your favorite media sources that report on data science topics?")
        
    
    option = st.selectbox(
     "Selectionnez le métiez sur lequel vous souhaiteriez obtenir plus d'information",
     listjob)
    
    option2 = st.selectbox(
     "Selectionnez le domaine dans lequel vous souhaiteriez avoir plus d'information",
     listQuestion)
    
    BarGraphPerc(QxToQuestion(option2[0:3].strip()), full_df.loc[(full_df['Q5'] == option)])
    
    
    option3 = st.selectbox(
     "x",
     listQuestion)
    
    option4 = st.selectbox(
     "y",
     (2,3,4))

    TopAnswersPerJob(QxToQuestion(option3[0:3].strip()), option4)

elif page==pages[2]:
    st.write("## Modélisation")