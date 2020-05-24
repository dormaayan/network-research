import randomGrid

def main():
	str = ''
	for i in range(0,2):
		s = randomGrid.simpleGrid(consider_coverage=False)
		str = str + '\n'
	print(str)


if __name__ == '__main__':
    main()