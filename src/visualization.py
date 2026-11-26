import pandas as pd
import matplotlib.pyplot as plt
from utils import OUTPUT_DIR, FIG_DIR, LABEL_NAMES, SPLIT_DIR
from sklearn.metrics import ConfusionMatrixDisplay



# 3 models comparison
def plot_model_comparison():
      test_results = pd.read_csv(OUTPUT_DIR/"test_results.csv")
      fig, ax = plt.subplots(figsize=(7, 4))
      bars = ax.bar(test_results['model'],test_results['macro_f1'],
                    color=["#B4B2A9", "#B4B2A9", "#185FA5"])
      
      ax.bar_label(bars, fmt="%.2f")        # writes each bar's value on top of it, 2 decimals
      ax.set_title("DistilBERT beats the TF-IDF baseline by 0.13 macro F1")   # title = the finding
      ax.set_ylabel("Macro F1 (test set)")  # axis label
      ax.set_ylim(0, 1)                     # y-axis from 0 to 1
      ax.spines[["top", "right"]].set_visible(False)   # removes the top and right borders

      fig.savefig(FIG_DIR / "model_comparison.png", dpi=200, bbox_inches="tight")   # saves the PNG
      plt.close(fig)                        # frees memory

      
# confusion matrix
def plot_confusion_matrix():
      preds = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
      fig, ax = plt.subplots(figsize=(5,4))

      ConfusionMatrixDisplay.from_predictions(
            preds["true"],          # rows = true class
            preds["predicted"],     # columns = predicted class
            display_labels=LABEL_NAMES,         # names instead of 0/1/2
            cmap="Blues",                 # darker = more sentences
            colorbar=False,               # the numbers are already in the cells
            ax=ax,                  # draw on our figure
      )
      n_wrong = (preds["true"] != preds["predicted"]).sum()
      ax.set_title(f"DistilBERT misclassifies {n_wrong} of {len(preds)} test sentences")

      fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
      plt.close(fig)


# category errors (based on 'misclassified_val_tagged.csv' file, where I manually created and labeled the categories 
# for mispredicted observations)
CATEGORY_NAMES = {
      "missing_ctxt":"Missing context",
      "plan":"Plans / announcements",
      "fluct_word": "Sentiment-laden words",
      "numbers": "Number comparisons",
      "debate": "Debatable label",
      "other": "Other",
}

def plot_error_categories():

      tagged = pd.read_csv(OUTPUT_DIR / 'misclassified_val_tagged.csv')   # 1. the tagged CSV in OUTPUT_DIR
      counts = tagged["category"].value_counts()      # 2. errors per category
      counts = counts.rename(CATEGORY_NAMES)          # 3. codes rename()-> display names
      counts = counts.sort_values(ascending=True)              # 4. ascending, so barh puts the largest on top


      fig, ax = plt.subplots(figsize=(7,4))
      bars = ax.barh(counts.index, counts.values, color="#185FA5")      # 5. y = names (the index), width = counts (the values)
      ax.bar_label(bars, padding=3)                                       # counts at the end of each bar
      
      ax.set_title("Errors split between data limits and specific model weaknesses")         # 6. the finding, from your counts
      ax.set_xlabel("Misclassified validation sentences")
      ax.spines[["top", "right"]].set_visible(False)

      fig.savefig(FIG_DIR / "error_categories.png", dpi=200, bbox_inches="tight")
      plt.close(fig)


# class balance
def plot_class_balance():
    # all 3,448 sentences = the three splits combined
    df = pd.concat([pd.read_csv(SPLIT_DIR / f"{s}.csv") for s in ["train", "val", "test"]])

    shares = df["label"].value_counts(normalize=True) * 100   # share of each label, in %
    shares = shares.sort_index()                              # order 0, 1, 2 = negative, neutral, positive

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(LABEL_NAMES, shares.values,
                  color=["#B4B2A9", "#185FA5", "#B4B2A9"])    # neutral highlighted
    ax.bar_label(bars, fmt="%.0f%%")                          # e.g. "62%"

    ax.set_title("Neutral makes up 62% of sentences, so accuracy alone is misleading")
    ax.set_ylabel("Share of sentences (%)")
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(FIG_DIR / "class_balance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)




def main():
      FIG_DIR.mkdir(exist_ok=True)
      plot_model_comparison()
      plot_confusion_matrix()
      plot_error_categories()
      plot_class_balance()



if __name__ == "__main__":
      main()