import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os


class PlotsDrawer:

	def __init__(self, folder_name='plots'):
		self.folder = folder_name
		if not os.path.exists(folder_name):
			os.makedirs(folder_name)

	def draw_plots(self, json_path):
		df = pd.read_json(json_path)

		json_name = os.path.basename(json_path).split('/')[-1].split('.')[0]
		save_path = f"{self.folder}/{json_name}"
		if not os.path.exists(save_path):
			os.makedirs(save_path)

		plots_paths = []

		# confusion matrix
		conf_matrix = pd.crosstab(df['gt_corners'], df['rb_corners'])

		green = sns.light_palette("seagreen", as_cmap=True)
		green.set_under('tomato')
		p1 = sns.heatmap(conf_matrix, vmin=1, cmap=green, cbar_kws={'extend': 'min'}).set_title('confusion matrix')

		p = save_path + "/" + "confusion_matrix.png"
		p1.get_figure().savefig(p)
		plots_paths.append(os.path.normpath(p))
		plt.clf()

		# rooms count
		p2 = sns.histplot(data=df, x="gt_corners")
		p = save_path + "/" + "rooms_count.png"
		p2.get_figure().savefig(p)
		plots_paths.append(os.path.normpath(p))
		plt.clf()

		# mean deviation
		p3 = sns.histplot(df, x='mean', hue='gt_corners').set_title('mean deviation')
		p = save_path + "/" + "mean_deviation.png"
		p3.get_figure().savefig(p)
		plots_paths.append(os.path.normpath(p))
		plt.clf()

		# max deviation
		p4 = sns.histplot(df, x='max', hue='gt_corners').set_title('max deviation')
		p = save_path + "/" + "max_deviation.png"
		p4.get_figure().savefig(p)
		plots_paths.append(os.path.normpath(p))
		plt.clf()

		# min deviation
		p5 = sns.histplot(df, x='max', hue='gt_corners').set_title('min deviation')
		p = save_path + "/" + "min_deviation.png"
		p5.get_figure().savefig(p)
		plots_paths.append(os.path.normpath(p))
		plt.clf()

		# min and max deviations
		p6 = sns.scatterplot(data=df, x="max", y="min", size="gt_corners", hue="gt_corners").set_title('min and max deviations')
		p = save_path + "/" + "min_max_deviations.png"
		p6.get_figure().savefig(p)
		plots_paths.append(os.path.normpath(p))
		plt.clf()

		# mean boxs
		p7 = sns.boxplot(df, x='gt_corners', y='mean')
		p = save_path + "/" + "mean_boxs.png"
		p7.get_figure().savefig(p)
		plots_paths.append(os.path.normpath(p))
		plt.clf()

		# column relations
		p8 = sns.pairplot(df[list(set(df.columns) - {'gt_corners', 'rb_corners'})])
		p = save_path + "/" + "column_relations.png"
		p8.savefig(p)
		plots_paths.append(os.path.normpath(p))
		plt.clf()

		return '\n'.join(plots_paths)


if __name__ == '__main__':
	print(PlotsDrawer().draw_plots('deviation.json'))
