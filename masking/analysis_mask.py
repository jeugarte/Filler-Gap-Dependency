import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_with_custom_colors(grouped_data, title, colors):
    if len(colors) < len(grouped_data):
        raise ValueError("Not enough colors provided for the number of bars")

    plt.figure(figsize=(12, 8))
    bars = plt.bar(grouped_data.index, grouped_data, color=colors[:len(grouped_data)])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), va='bottom', ha='center', fontsize=20)

    plt.title(title)
    plt.xlabel('NLTK Tag')
    plt.ylabel('Normalized Probability')
    plt.yticks(np.arange(0, max(grouped_data)+0.05, step=0.05))
    plt.xticks(rotation=45, fontsize=16)

    plt.show()


def plot_token_distribution(grouped_data, title, colors):
    if len(colors) < len(grouped_data):
        raise ValueError("Not enough colors provided for the number of bars")

    grouped_data = grouped_data.sort_values(ascending=False)

    plt.figure(figsize=(15, 8)) 
    bars = plt.bar(grouped_data.index, grouped_data, color=colors[:len(grouped_data)])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3), va='bottom', ha='center', fontsize=16)

    plt.title(title)
    plt.xlabel('Predicted Token')
    plt.ylabel('Normalized Probability')

    max_prob = max(grouped_data)
    plt.yticks(np.arange(0, max(grouped_data)+0.05, step=0.05))

    plt.xticks(rotation=45, fontsize=10) 

    plt.show()





df = pd.read_csv('./experiments/wh_adjunct_mask/analysis/compiled_data.tsv', delimiter='\t')

df_obj_gap_n = df[df['obj_gap'] == 'n']
df_obj_gap_y = df[df['obj_gap'] == 'y']

# grouped_n = df_obj_gap_n.groupby('spacy_pos_tag')['probability'].sum()
# grouped_y = df_obj_gap_y.groupby('spacy_pos_tag')['probability'].sum()

grouped_n = df_obj_gap_n.groupby('nltk_pos_tag')['probability'].sum()
grouped_y = df_obj_gap_y.groupby('nltk_pos_tag')['probability'].sum()

grouped_n /= grouped_n.sum()
grouped_y /= grouped_y.sum()

grouped_n = grouped_n[grouped_n >= 0.01]
grouped_y = grouped_y[grouped_y >= 0.01]

grouped_n = grouped_n.sort_values(ascending=False)
grouped_y = grouped_y.sort_values(ascending=False)

salmon_rgb = (250/255, 128/255, 114/255)
dark_blue_rgb = (0/255, 0/255, 139/255)

dark_blue_shades = [
    "#00008B",  
    
    "#4169E1",  
    "#1E90FF",  
    "#87CEEB"   
]

# red_shades = [
#     "#8B0000",  
#     "#B22222", 
#     "#DC143C", 
#     "#FF4500", 
#     "#FF6347"  
# ]

red_shades = [
    "#8B0000",  
    "#A52A2A",  
    "#B22222",  
    "#CD5C5C",  
    "#DC143C",  
    "#FF0000",  
    "#FF4500",  
    "#FF6347", 
    "#FF7F50", 
    "#FA8072"   
]
plot_with_custom_colors(grouped_n, 'Syntactic Category in Filler Position w/o Object Gap', dark_blue_shades)
plot_with_custom_colors(grouped_y, 'Syntactic Category in Filler Position w/ Object Gap', red_shades)



#TOKEN PROBABILITIES

df_obj_gap_n = df[df['obj_gap'] == 'n']
df_obj_gap_y = df[df['obj_gap'] == 'y']

grouped_tokens_n = df_obj_gap_n.groupby('token')['probability'].sum()
grouped_tokens_y = df_obj_gap_y.groupby('token')['probability'].sum()


grouped_tokens_n /= grouped_tokens_n.sum()
grouped_tokens_y /= grouped_tokens_y.sum()

grouped_tokens_n = grouped_tokens_n[grouped_tokens_n >= 0.01]
grouped_tokens_y = grouped_tokens_y[grouped_tokens_y >= 0.01]

grouped_tokens_n = grouped_tokens_n.sort_values(ascending=False)
grouped_tokens_y = grouped_tokens_y.sort_values(ascending=False)


green_shades = [
    "#004d00", 
    "#006600",  
    "#008000", 
    "#009900",
    "#00b300",
    "#00cc00", 
    "#00e600",
    "#00ff00"  
]


turquoise_shades = [
    "#033d3d", 
    "#046666",
    "#058f8f",  
    "#06b8b8",
    "#07e1e1",
    "#04dede",  
    "#00d2d2",
    "#00c6c6",
    "#00bab9",
    "#00aeae",
    "#00a2a2",  
    "#009696",
    "#008a8a",  
    "#007e7e",
    "#007272",  
    "#006666"
]


plot_token_distribution(grouped_tokens_n, 'Probability of Predicted Token in Filler Position w/o Object Gap', green_shades)
plot_token_distribution(grouped_tokens_y, 'Probability of Predicted Token in Filler Position w/ Object Gap', turquoise_shades)

