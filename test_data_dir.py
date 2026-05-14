import os

print(f"Contents of /Users/ege/Desktop/346_project/data:")
print(os.listdir("/Users/ege/Desktop/346_project/data"))
for root, dirs, files in os.walk("/Users/ege/Desktop/346_project/data"):
    for file in files:
        print(os.path.join(root, file))
