"""
    walrus operator
    assigns values to variables as part of a larger expression
"""
favoriteActivities= list()
while activity := input("Enter your favorite activity: ") != "exit" :
    favoriteActivities.append(activity)