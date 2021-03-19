def find_attribute(code, attribute, listOfTuples):
    for item in listOfTuples:
        if item[0] == code and item[1] == attribute:
            return item[2]
    return "error"

# import tracemalloc
# tracemalloc.start()

brands = ["2kbrands", "Anello", "Anna Riska", "Bengi", "Benzi", "Ezzo", "Fidelio Home", "Guy Laroche", "Inart", "KENTIA",
"Makis Tselios", "MAW", "pakketo", "pakoworld", "PANTONE", "PAUL FRANK", "RAFEVI", "Rythmos", "Saint Clair", "Viopros"]

f = open("codes_init.txt", "r")

codes_init = []

for item in f:
    item = item.strip()
    codes_init.append(item)

f.close()

codes = []
attributes = []
values = []

f = open("codes.txt", "r")

for item in f:
    item = item.strip()
    codes.append(item)

f.close()

f = open("attributes.txt", "r")

for item in f:
    item = item.strip()
    attributes.append(item)

f.close()

f = open("values.txt", "r")

for item in f:
    item = item.strip()
    values.append(item)

f.close()

listOfTuples = []

for code, attribute, value in zip(codes, attributes, values):
    listOfTuples.append((code, attribute, value))

# for item in values:
#     print(item)

for code in codes_init:
    value = find_attribute(code, "Υλικό", listOfTuples)
    print(code, value)

# current, peak = tracemalloc.get_traced_memory()
# print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
# tracemalloc.stop()