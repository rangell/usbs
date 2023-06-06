import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from IPython import embed

plt.rcParams.update({'font.size': 8})

with open(f'sims_summary_df_0_90_G67.pkl', 'rb') as f:
    df_90 = pickle.load(f)
    df_90['warm-start percentage'] = '90%'

with open(f'sims_summary_df_0_95_G67.pkl', 'rb') as f:
    df_95 = pickle.load(f)
    df_95['warm-start percentage'] = '95%'

with open(f'sims_summary_df_0_99_G67.pkl', 'rb') as f:
    df_99 = pickle.load(f)
    df_99['warm-start percentage'] = '99%'

merged_df = pd.concat([df_90, df_95, df_99])

tol = 0.01
mask_90 = (df_90['feasibility'] < tol) * (df_90['objective residual'] < tol)
mask_95 = (df_95['feasibility'] < tol) * (df_95['objective residual'] < tol)
mask_99 = (df_99['feasibility'] < tol) * (df_99['objective residual'] < tol)

masked_df_90 = df_90[mask_90]
masked_df_95 = df_95[mask_95]
masked_df_99 = df_99[mask_99]

masked_df = pd.concat([masked_df_90, masked_df_95, masked_df_99])

solvers = sorted(list(set(masked_df['solver'])))
warm_start_percentages = sorted(list(set(masked_df['warm-start percentage'])))

df_rows = []
for solver in solvers:
    for percentage in warm_start_percentages:
        try:
            df_rows.append(masked_df[(masked_df['solver'] == solver) * (masked_df['warm-start percentage'] == percentage)].iloc[0])
        except:
            df_rows.append(merged_df[(merged_df['solver'] == solver) * (merged_df['warm-start percentage'] == percentage)].iloc[-1])


pd.set_option("display.precision", 8)
selected_df = pd.DataFrame(df_rows)

ax = sns.barplot(data=selected_df, x="warm-start percentage", y="time (sec)", hue="solver")
for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=7, color='black', xytext=(0, 5),
                textcoords='offset points')
_ = ax.set_ylim(0,1000) #To make space for the annotations
plt.title('G67 :: Time to 0.01 approx solution')
#plt.show()
plt.savefig(f'G67-time-until-0_01.png')

exit()

#palette = sns.color_palette('husl', 8)[5:8]

xs = ['time (sec)', 'iteration']
ys = ['objective residual', 'feasibility', 'weight of cut']

for x in xs:
    for y in ys:
        g = sns.lineplot(
                data=sims_summary_df,
                x=x,
                y=y,
                hue_order=sorted(list(set(sims_summary_df["solver"]))),
                hue='solver',
                errorbar=None,
                style_order=sorted(list(set(sims_summary_df["solver"]))),
                style='solver',
                markers=False,
                #palette=palette,
                linewidth=2,
                dashes=False
        )
        plt.title('G67 :: 90% warm-start')
        [l.set_linewidth(2) for l in plt.legend().get_lines()]
        #if dataset != 'pubmed':
        #    g.legend_.remove()
        #plt.xlabel('')
        #plt.ylabel('')
        plt.xscale('log')
        if y != 'weight of cut':
            plt.yscale('log')
        x_ = x.replace(' ', '_')
        y_ = y.replace(' ', '_')
        plt.savefig(f'0_90_G67_{x_}-vs-{y_}.png')
        plt.clf()

exit()


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

