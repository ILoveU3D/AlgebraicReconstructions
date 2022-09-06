def projection2Astra(projection):
    x, y, z = projection.shape
    return projection.reshape([z,y,x]).permute(2,0,1).reshape([z,y,x])

def astra2Projection(astra):
    z, y, x = astra.shape
    return astra.permute(1,2,0).reshape([z,y,x])