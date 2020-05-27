from rouge import FilesRouge

files_rouge = FilesRouge()
scores = files_rouge.get_scores('C:\\Users\\thisi\\Desktop\\sys.txt', 'C:\\Users\\thisi\\Desktop\\model.txt', avg=True)

print(scores)