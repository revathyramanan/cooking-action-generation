from infer import CookGenModelWrapper
import gc

if __name__ == '__main__':
    
    gc.collect()

    model = CookGenModelWrapper()
    inference = model.infer("Boil the chicken upto 100 degrees celcius")
    print (inference)
