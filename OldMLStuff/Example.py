
def reclassify():
    final_data = []
    for scrub_weight in range(6, 9):
        for agri_weight in range(4, 7):
            previous_data = f"1 1;2 1;3 2;4 {agri_weight};5 5;6 8;7 9;8 10;9 {scrub_weight};10 6;11 2;12 2"
            final_data.append(previous_data)

    return final_data

data = reclassify()
print(data)

    