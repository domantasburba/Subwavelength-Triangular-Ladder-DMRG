# module Utility

import Dates
# using Suppressor
using Printf
using PyCall
import PyPlot as plt

# export get_fig_ax, set_plot_defaults, save_plot

function sparse_display(A)
    display(real.(Matrix(remove_insignificant_terms(A))))
end # function

function remove_insignificant_terms!(A, reltol=1e-6)
    max_term = maximum(abs.(A))
    abstol = max_term * reltol
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            if abs(A[i, j]) < abstol
                A[i, j] = 0.0
            end # if
        end # for
    end # for
end # function

function remove_insignificant_terms(A, reltol=1e-6)
    B = deepcopy(A)
    remove_insignificant_terms!(B, reltol)

    return B
end # function

function save_animation(anim, name, save_dir; fps=30, file_type="gif")
    save_path = @sprintf "%s/Animations/%s/%s" save_dir Dates.today() name
    name_string = @sprintf "%s!%s.%s" name Dates.format(Dates.now(), "Y-mm-dd!HH:MM:SS") file_type
    mkpath(save_path)
    file_path = @sprintf "%s/%s" save_path name_string

    if file_type == "gif"
        # anim.save(file_path, fps=fps, writer="imagemagick")
        anim.save(file_path, writer="imagemagick")
    elseif file_type == "mp4"
        anim.save(file_path, fps=fps, writer="ffmpeg")
    else
        println("WARNING: Unsupported file type.")
    end # if
end # function

function save_parameter_TXT(save_path)
    script_dir = dirname(save_path)
    save_dir_path = string(script_dir, "/Parameters/", Dates.today())
    mkpath(save_dir_path)

    date_now = Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS")
    script_name = save_path[length(script_dir)+2:end]
    parameter_TXT_path = string(save_dir_path, "/", script_name, "!", date_now, ".txt")

    # println(script_name)
    # script = open(save_path, "r")
    script_text = strip.(readlines(save_path))  # strip removes whitespace from string.
    parameter_TXT = open(parameter_TXT_path, "w")

    # TODO readuntil function may be simpler.
    idx_start = findall(x -> x == "# PARAMETERS", script_text)[1]
    idx_end = findall(x -> x == "# END OF PARAMETERS", script_text)[1]
    for idx_line in idx_start:idx_end
        write(parameter_TXT, string(script_text[idx_line], "\n"))
    end # for

    close(parameter_TXT)
end # function

function save_script_TXT(script_path)
    script_dir = dirname(script_path)
    save_dir_path = string(script_dir, "/Script_Text/", Dates.today())
    mkpath(save_dir_path)

    date_now = Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS")
    script_name = script_path[length(script_dir)+2:end]
    script_TXT_path = string(save_dir_path, "/", script_name, "!", date_now, ".txt")

    # println(script_name)
    # script = open(script_path, "r")
    # script_text = strip.(readlines(script_path))  # strip removes whitespace from string.
    script_text = readlines(script_path)
    script_TXT = open(script_TXT_path, "w")

    for idx_line in 1:length(script_text)
        write(script_TXT, string(script_text[idx_line], "\n"))
    end # for

    close(script_TXT)
end # function

# Made with GPT-4
# WARNING: This implementation is slow (more allocations & longer compile times).
function save_stdout_TXT(code::Function, script_path)
    script_dir = dirname(script_path)
    save_dir_path = string(script_dir, "/stdout/", Dates.today())
    mkpath(save_dir_path)

    date_now = Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS")
    script_name = script_path[length(script_dir)+2:end]
    stdout_TXT_path = string(save_dir_path, "/", script_name, "!", date_now, ".txt")

    # Save the current standard output
    original_stdout = stdout

    # Create a temporary file
    tmp_path, tmp_io = mktemp()

    # Redirect the standard output to the temporary file
    redirect_stdout(tmp_io) do
        try
            # Execute the given code
            code()
        catch e
            # In case of an error, restore the original standard output and rethrow the error
            redirect_stdout(original_stdout)
            close(tmp_io)
            rm(tmp_path)
            throw(e)
        end # try
    end # do

    # Restore the original standard output
    redirect_stdout(original_stdout)
    close(tmp_io)

    # Copy the temporary file content to the output file and print it to the terminal
    open(stdout_TXT_path, "w") do out_file
        open(tmp_path, "r") do in_file
            for line in eachline(in_file)
                println(line)
                println(out_file, line)
            end # for
        end # do
    end # do

    # Remove the temporary file
    rm(tmp_path)
end # function

# end # module
