# kwargs : Key Word Arguments
# ** in a function declaration means collect all keyword arguments into a dictionary
# named kwargs.
# And for the callee 
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Call with any number of keyword arguments
print_info(name="Dan", age=50, city="Stockholm")

dict = {"name":"Dan", "age":50, "city":"Stockholm"}
print_info(**dict)  # Unpack the dictionary into keyword arguments
