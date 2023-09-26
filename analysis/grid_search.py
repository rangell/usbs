import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from IPython import embed


if __name__ == "__main__":

    #with open("sims_summary_df_chr12a.pkl", "rb") as f:
    #    whole_df = pickle.load(f)
    #with open("sims_summary_df_ulysses16.pkl", "rb") as f:
    #    whole_df = pickle.load(f)
    with open("sims_summary_df_esc32a.pkl", "rb") as f:
        whole_df = pickle.load(f)

    embed()
    exit()

    slim_df = whole_df.groupby(
        ["k_curr",
         "k_past",
         "trace_factor",
         "rho",
         "beta",
         "warm_start_strategy"])["callback value"].min().reset_index()

    plt.rcParams.update({'font.size': 8})

    sns.clustermap(slim_df.pivot_table(
        values=["callback value"],
        index=["trace_factor", "rho", "beta"],
        columns=["k_curr", "k_past", "warm_start_strategy"]
    ), annot=True)

    plt.gcf().set_size_inches(18.5, 10.5)
    plt.show()
    
    embed()
    exit()