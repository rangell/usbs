import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython import embed

plt.rcParams.update({'font.size': 16})

dataset = 'pubmed'
with open(f'{dataset}_sims_summary_df.pkl', 'rb') as f:
    sims_summary_df = pickle.load(f)

#palette = sns.color_palette('husl', 8)[5:8]

g = sns.lineplot(
        data=sims_summary_df,
        x='CU',
        y='F1',
        hue_order=['ECC', 'MLCL-single', 'MLCL-batched'],
        hue='constraint_type',
        ci=None,
        style_order=['ECC', 'MLCL-single', 'MLCL-batched'],
        style='constraint_type',
        markers=False,
        #palette=palette,
        linewidth=3,
        dashes=False
)
g.legend_.set_title(None)
[l.set_linewidth(3) for l in plt.legend().get_lines()]
if dataset != 'pubmed':
    g.legend_.remove()
plt.xlabel('')
plt.ylabel('')
plt.savefig(f'{dataset}_cu-vs-f1.png')

plt.clf()

g = sns.lineplot(
        data=sims_summary_df,
        x='CU',
        y='Rand Idx',
        hue_order=['ECC', 'MLCL-single', 'MLCL-batched'],
        hue='constraint_type',
        ci=None,
        style_order=['ECC', 'MLCL-single', 'MLCL-batched'],
        style='constraint_type',
        markers=False,
        #palette=palette,
        linewidth=3,
        dashes=False
)
g.legend_.set_title(None)
[l.set_linewidth(3) for l in plt.legend().get_lines()]
if dataset != 'pubmed':
    g.legend_.remove()
plt.xlabel('')
plt.ylabel('')
plt.savefig(f'{dataset}_cu-vs-rand_idx.png')

plt.clf()

g = sns.lineplot(
        data=sims_summary_df,
        x='CU',
        y='FMC',
        hue_order=['ECC', 'MLCL-single', 'MLCL-batched'],
        hue='constraint_type',
        ci=None,
        style_order=['ECC', 'MLCL-single', 'MLCL-batched'],
        style='constraint_type',
        markers=False,
        #palette=palette,
        linewidth=3,
        dashes=False
)
g.legend_.set_title(None)
[l.set_linewidth(3) for l in plt.legend().get_lines()]
if dataset != 'pubmed':
    g.legend_.remove()
plt.xlabel('')
plt.ylabel('')
plt.savefig(f'{dataset}_cu-vs-fmc.png')

#sns.lineplot(data=sims_summary_df, x='# constraints', y='Rand Idx', hue='constraint_type')
#sns.lineplot(data=sims_summary_df, x='CU', y='FMC', hue='constraint_type')
#plt.show()

