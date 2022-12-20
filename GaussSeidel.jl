import LinearAlgebra
import BandedMatrices
import SparseArrays

matb(n) = ones(n,1)

function matA(n) 
    LinearAlgebra.diagm(0 => 2*ones(n), -1 => -1*ones(n-1), 1 => -1*ones(n-1))
end


function matD(A)
    n, n = size(A)
    H = SparseArrays.spzeros(n,n)
    for i = 1:n
        H[i,i] = A[i,i]
    end
    return H
end


function gsrelax(n,w)
    e = 10^-8
    A = matA(n)*(n^2) #to account for 1/h
    b = matb(n)
    D = matD(A)
    U = LinearAlgebra.triu(A) - D
    L = LinearAlgebra.tril(A) - D
    M = D/w+L
    N = (1-w)/w*D - U
    x = zeros(n,1)
    iter = 0
    while (LinearAlgebra.norm(A*x-b)>=e*LinearAlgebra.norm(b))
        x = M\(N*x+b)
        iter+=1
        println(iter)
    end
    return x
end

G = @time gsrelax(50,1.8)
println(G)

