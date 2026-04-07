function op_to_lattice_op(op, M::Int, i::Int)
    @assert 1 <= i <= M
    @assert size(op)[1] == size(op)[2]

    op_i = [one(eltype(op))]
    id_op = id(size(op)[1])
    for j in 1:M
        if i == j
            op_i = kron(op_i, op)
        else
            op_i = kron(op_i, id_op)
        end # if
    end # for

    return op_i
end # function

function op_to_lattice_ops(op, M::Int)
    return [op_to_lattice_op(op, M, i) for i in 1:M]
end # function

function lattice_ops_to_collective_op(ops::Vector)
    return sum(ops)
end # function

function op_to_collective_op(op, M::Int)
    return lattice_ops_to_collective_op(op_to_lattice_ops(op, M))
end # function
