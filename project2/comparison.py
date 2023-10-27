with open('build/result_standard1.txt', 'r') as file1, open('build/result1.txt', 'r') as file2:
    for line1, line2 in zip(file1, file2):
        if line1 != line2:
            print("The file1 are not the same.")
            break
    else:
        print("file1 same.")


with open('build/result_standard2.txt', 'r') as file1, open('build/result2.txt', 'r') as file2:
    for line1, line2 in zip(file1, file2):
        if line1 != line2:
            print("The file2 are not the same.")
            break
    else:
        print("file2 same.")
        