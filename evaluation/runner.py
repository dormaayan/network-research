import randomGrid
import csv   


def main():
	#str = ''
	#for i in range(0,2):
	s = randomGrid.simpleGrid(consider_coverage=False)
	with open(r'res.csv', 'a') as f:
	    writer = csv.writer(f)
	    writer.writerow(s)
	#	str = str + '\n'
	#print(str)


if __name__ == '__main__':
    main()