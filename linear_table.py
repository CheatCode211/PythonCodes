def main ():
    scores = [ 65 , 90 , 56 , 70 , 40 , 95 ]
    length = len( scores)
    for i in range( length := len(scores)):
        print ( scores[ i], end="," if i < length-1 else "" )
if __name__ == "__main__" :
    main()