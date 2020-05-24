import randomGrid
import csv   


def main():
	s = randomGrid.simpleGrid(consider_coverage=False)
	with open(r'res.csv', 'a') as f:
	    writer = csv.writer(f, delimiter = ',')
	    writer.writerow(s)


if __name__ == '__main__':
    main()