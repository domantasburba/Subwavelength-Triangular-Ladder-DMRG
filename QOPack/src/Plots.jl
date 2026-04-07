"""Used to create flexible single graph functions, which can easily be used
to create composite plots containing multiple graphs.\n
fig_ax is either None or a tuple of matplotlib's figure and axis objects.
If fig_ax is None, a single graph will be plotted. If fig_ax is given, the
graph will be added to the composite plot.\n
fig_name is either None or a string. fig_name only takes effect if fig_ax is
None, i.e., a single graph is plotted. fig_name sets the figure name."""
function get_fig_ax(fig_ax=nothing; fig_name=nothing)
    if fig_ax === nothing
        if fig_name === nothing
            fig = plt.figure()
        else
            fig = plt.figure(fig_name)
        end # if
        ax = plt.axes()
    else
        fig, ax = fig_ax
    end # if
    
    return fig, ax
end # function

function set_plot_defaults(fig, ax::PyCall.PyObject; addGrid=true)
    # rcParams = plt.PyDict(plt.matplotlib."rcParams")
    # rcParams["font.size"] = 11
    # fig.tight_layout()
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    if addGrid
        ax.grid()
    end # if
end # function

function set_plot_defaults(fig, ax::Vector{PyCall.PyObject}; addGrid=true)
    # rcParams = plt.PyDict(plt.matplotlib."rcParams")
    # rcParams["font.size"] = 11
    fig.tight_layout()
    ax_1D = view(ax, :)
    [ax_i.minorticks_on() for ax_i in ax_1D]
    [ax_i.tick_params(axis="both", which="both", direction="in") for ax_i in ax_1D]
    [ax_i.xaxis.set_ticks_position("both") for ax_i in ax_1D]
    [ax_i.yaxis.set_ticks_position("both") for ax_i in ax_1D]
    if addGrid
        [ax_i.grid() for ax_i in ax_1D]
    end # if
end # function

function set_plot_defaults(fig, ax::Array{PyCall.PyObject}; addGrid=true)
    # rcParams = plt.PyDict(plt.matplotlib."rcParams")
    # rcParams["font.size"] = 11
    fig.tight_layout()
    ax_1D = view(ax, :)
    [ax_i.minorticks_on() for ax_i in ax_1D]
    [ax_i.tick_params(axis="both", which="both", direction="in") for ax_i in ax_1D]
    [ax_i.xaxis.set_ticks_position("both") for ax_i in ax_1D]
    [ax_i.yaxis.set_ticks_position("both") for ax_i in ax_1D]
    if addGrid
        [ax_i.grid() for ax_i in ax_1D]
    end # if
end # function

function save_plot(name, save_dir; saveDirectly=false, addTime=true, file_type="png", dpi=400)
    # if save_path === nothing
    #     save_path = string(@__DIR__, "/Graphs/", Dates.today())
    #     # save_path = string(pwd(), "/Graphs/", Dates.today())
    # end # if

    if saveDirectly
        save_path = string(save_dir)
    else
        if findfirst('!', name) == nothing
            dir_name = name
        else
            dir_name = name[1:findfirst('!', name)-1]
        end # if

        # save_path = string(save_dir, "/Graphs/", Dates.today())
        save_path = string(save_dir, "/Graphs/", Dates.today(), "/", dir_name)
    end # if
    mkpath(save_path)

    if addTime
        name_string = string("/", name, "!", Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"), ".", file_type)
    else
        name_string = string("/", name, ".", file_type)
    end # if
    # name_string = string("/", name, "!", Dates.format(Dates.now(), "Y-mm-dd!HH.MM.SS"), ".", "svg")

    # plt.savefig(string(save_path, name_string), dpi=dpi)
    # plt.savefig(string(save_path, name_string), format="eps", dpi=dpi)
    # plt.savefig(string(save_path, name_string), format="svg", dpi=dpi)
    if file_type == "svg"
        plt.savefig(string(save_path, name_string), format="svg", dpi=dpi)
    else
        plt.savefig(string(save_path, name_string))
        # plt.savefig(string(save_path, name_string), dpi=dpi)
    end # if
end # function

function plot_y_vs_x(x_arr, y_arr, save_path; fig_size=nothing, x_scale="linear", y_scale="linear",
    x_lims=nothing, y_lims=nothing, x_label=raw"$x$", y_label=raw"$y$", ax_title=nothing,
    color="green", marker=nothing, saveDirectly=true, fig_ax=nothing, file_name="y_vs_x")
    fig, ax = get_fig_ax(fig_ax)
    # if fig_ax == nothing
    #     fig.set_size_inches(8, 4)
    # end # if

    if fig_ax == nothing
        if fig_size == nothing
            fig_size = (9, 6)
        end # if

        fig.set_size_inches(fig_size...)
    end # if
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ax_title != nothing
        ax.set_title(ax_title)
    end # if
    ax.set_xlim(x_arr[1], x_arr[end])
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    # ax.set_ylim(0.01, 1.5)
    if x_lims != nothing
        ax.set_xlim(x_lims...)
    end # if
    if y_lims != nothing
        ax.set_ylim(y_lims...)
    end # if
    if marker == nothing
        ax.plot(x_arr, y_arr, color=color)
    else
        ax.plot(x_arr, y_arr, color=color, marker=marker)
    end # if

    if fig_ax == nothing
        set_plot_defaults(fig, ax)
        save_plot(file_name, save_path; saveDirectly=saveDirectly)
        plt.show()
    end # if
end # function

function plot_ys_vs_x(x_arr, y_arr_vec, save_path; fig_size=nothing, x_scale="linear", y_scale="linear",
    x_lims=nothing, y_lims=nothing, x_label=raw"$x$", y_label=raw"$y$", ax_title=nothing,
    color_vec=nothing, ls_vec=nothing, saveDirectly=true,
    fig_ax=nothing, file_name="ys_vs_x")
    N_y = length(y_arr_vec)

    fig, ax = get_fig_ax(fig_ax)
    # if fig_ax == nothing
    #     fig.set_size_inches(8, 4)
    # end # if

    if fig_ax == nothing
        if fig_size == nothing
            fig_size = (9, 6)
        end # if

        fig.set_size_inches(fig_size...)
    end # if
    if color_vec == nothing
        # color_vec = ["green" for i in 1:N_y]
        color_vec = ["blue", "purple", "red", "green", "black", "magenta"]
    end # if
    if ls_vec == nothing
        ls_vec = ["solid", "dashed", "dotted", "dashdot", "long dash with offset"]
    end # if

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ax_title != nothing
        ax.set_title(ax_title)
    end # if
    ax.set_xlim(x_arr[1], x_arr[end])
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    # ax.set_ylim(0.01, 1.5)
    if x_lims != nothing
        ax.set_xlim(x_lims...)
    end # if
    if y_lims != nothing
        ax.set_ylim(y_lims...)
    end # if
    for (i, y_arr) in enumerate(y_arr_vec)
        ax.plot(x_arr, y_arr, color=color_vec[i], ls=ls_vec[i])
    end # if

    if fig_ax == nothing
        set_plot_defaults(fig, ax)
        save_plot(file_name, save_path; saveDirectly=saveDirectly)
        plt.show()
    end # if
end # function

function plotN_ys_vs_x(x_arr, y_arr_vec, save_path; ax_title=nothing,
    x_scale_vec=nothing, y_scale_vec=nothing,
    x_lims_vec=nothing, y_lims_vec=nothing,
    x_label_vec=nothing, y_label_vec=nothing, colors=nothing,
    marker=nothing, saveDirectly=true, file_name="N_ys_vs_x")
    N_y = length(y_arr_vec)

    if x_scale_vec == nothing
        x_scale_vec = ["linear" for i in 1:N_y]
    end # if
    if y_scale_vec == nothing
        y_scale_vec = ["linear" for i in 1:N_y]
    end # if
    if x_lims_vec == nothing
        x_lims_vec = [nothing for i in 1:N_y]
    end # if
    if y_lims_vec == nothing
        y_lims_vec = [nothing for i in 1:N_y]
    end # if
    if x_label_vec == nothing
        x_label_vec = vcat(["" for i in 1:(N_y-1)], [raw"$x$"])
    end # if
    if y_label_vec == nothing
        y_label_vec = [raw"$y$" for i in 1:N_y]
    end # if
    if colors == nothing
        colors = ["blue", "red", "purple", "green", "magenta", "black", "gray", "crimson", "cyan"]
    end # if

    fig, ax = plt.subplots(N_y, 1)
    fig.set_size_inches(8, 4*N_y+1)
    plt.subplots_adjust(left=0.15, bottom=0.12, top=0.95, wspace=0.1, hspace=0.25, right=0.97)

    if ax_title != nothing
        ax[1].set_title(ax_title)
    end # if
    for (i, y_arr) in enumerate(y_arr_vec)
        plot_y_vs_x(x_arr, y_arr, save_path; x_scale=x_scale_vec[i], y_scale=y_scale_vec[i], x_lims=x_lims_vec[i], y_lims=y_lims_vec[i], x_label=x_label_vec[i], y_label=y_label_vec[i], color=colors[i], marker=marker, fig_ax=(fig, ax[i]))
    end # for

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.2)
    save_plot(file_name, save_path; saveDirectly=saveDirectly)
    plt.show()
end # function

function plotN_grid_ys_vs_x(x_arr, y_arr_vec, save_path; N_cols=3,
    x_scale_vec=nothing, y_scale_vec=nothing, x_lims_vec=nothing,
    y_lims_vec=nothing, x_label_vec=nothing, y_label_vec=nothing,
    colors=nothing, saveDirectly=true, file_name="N_grid_ys_vs_x")
    N_rows = (length(y_arr_vec) - 1) ÷ N_cols + 1
    N_y = length(y_arr_vec)

    if x_scale_vec == nothing
        x_scale_vec = ["linear" for i in 1:N_y]
    end # if
    if y_scale_vec == nothing
        y_scale_vec = ["linear" for i in 1:N_y]
    end # if
    if x_lims_vec == nothing
        x_lims_vec = [nothing for i in 1:N_y]
    end # if
    if y_lims_vec == nothing
        y_lims_vec = [nothing for i in 1:N_y]
    end # if
    if x_label_vec == nothing
        x_label_vec = vcat(["" for i in 1:(N_y-1)], [raw"$x$"])
    end # if
    if y_label_vec == nothing
        y_label_vec = [raw"$y$" for i in 1:N_y]
    end # if
    if colors == nothing
        colors = ["blue", "red", "purple", "green", "magenta", "black"]
    end # if

    fig, ax = plt.subplots(N_rows, N_cols)
    fig.set_size_inches(7*N_rows+2, 4*N_cols+2)
    plt.subplots_adjust(left=0.15, bottom=0.12, top=0.95, wspace=0.1, hspace=0.25, right=0.97)

    for (i, y_arr) in enumerate(y_arr_vec)
        idx_row = (i-1) ÷ N_cols + 1
        idx_col = (i-1) % N_cols + 1

        plot_y_vs_x(x_arr, y_arr, save_path; x_scale=x_scale_vec[i], y_scale=y_scale_vec[i], x_lims=x_lims_vec[i], y_lims=y_lims_vec[i], x_label=x_label_vec[i], y_label=y_label_vec[i], color=colors[i], fig_ax=(fig, ax[idx_row, idx_col]))
    end # for

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.2)
    save_plot(file_name, save_path; saveDirectly=saveDirectly)
    plt.show()
end # function

function plot_u_vs_xy(x_arr, y_arr, u_arr, save_path; fig_size=nothing, x_name=raw"$x$", y_name=raw"$y$", x_lim=nothing, y_lim=nothing,
    ax_title=nothing, cbar_label=nothing, marker_xs=nothing, marker_ys=nothing, marker_s=5.0, marker_c="w",
    marker_type="o", cmap="viridis", vlim=nothing, saveDirectly=true, fig_ax=nothing, file_name="u_vs_xy")
    fig, ax = get_fig_ax(fig_ax)

    if fig_ax == nothing
        if fig_size == nothing
            fig_size = (9, 6)
        end # if

        fig.set_size_inches(fig_size...)
    end # if
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    if x_lim == nothing
        x_lim = [x_arr[1], x_arr[end]]
    end # if
    ax.set_xlim(x_lim...)
    if y_lim == nothing
        y_lim = [y_arr[1], y_arr[end]]
    end # if
    ax.set_ylim(y_lim...)
    if ax_title != nothing
        ax.set_title(ax_title)
    end # if
    if vlim == nothing
        pmesh = ax.pcolormesh(x_arr, y_arr, transpose(u_arr), cmap=plt.get_cmap(cmap), shading="nearest")
    else
        pmesh = ax.pcolormesh(x_arr, y_arr, transpose(u_arr), cmap=plt.get_cmap(cmap), vmin=vlim[1], vmax=vlim[2], shading="nearest")
    end # if
    if (marker_xs != nothing) && (marker_ys != nothing)
        ax.scatter(marker_xs, marker_ys, s=marker_s, c=marker_c, marker=marker_type)
    end # if
    if cbar_label == nothing
        cbar = fig.colorbar(pmesh, ax=ax, location="right", aspect=30)
    else
        cbar = fig.colorbar(pmesh, ax=ax, location="right", aspect=30, label=cbar_label)
    end # if
    # cbar.ax.xaxis.label.set_size(12)
    cbar.ax.tick_params(which="both", length=2, direction="in")
    cbar.ax.xaxis.set_ticks_position("both")

    if fig_ax == nothing
        set_plot_defaults(fig, ax, addGrid=false)
        save_plot(file_name, save_path; saveDirectly=saveDirectly)
        plt.show()
    end # if
end # function

function plot_u_w_streamlines_vs_xy(x_arr, y_arr, u_arr, vx_arr, vy_arr, save_path; x_name=raw"$x$", y_name=raw"$y$", x_lim=nothing, y_lim=nothing,
    ax_title=nothing, cbar_label=nothing, streamline_color="w", streamline_density=1, streamline_lw=2, streamline_start_points=nothing,
    marker_xs=nothing, marker_ys=nothing, marker_s=5.0, marker_c="w", marker_type="o", cmap="viridis",
    vlim=nothing, saveDirectly=true, fig_ax=nothing, file_name="u_w_streamlines_vs_xy")
    fig, ax = get_fig_ax(fig_ax)

    if fig_ax == nothing
        fig.set_size_inches(9, 6)
    end # if
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    if x_lim == nothing
        x_lim = [x_arr[1], x_arr[end]]
    end # if
    ax.set_xlim(x_lim...)
    if y_lim == nothing
        y_lim = [y_arr[1], y_arr[end]]
    end # if
    ax.set_ylim(y_lim...)
    if ax_title != nothing
        ax.set_title(ax_title)
    end # if
    if vlim == nothing
        pmesh = ax.pcolormesh(x_arr, y_arr, transpose(u_arr), cmap=plt.get_cmap(cmap), shading="nearest")
    else
        pmesh = ax.pcolormesh(x_arr, y_arr, transpose(u_arr), cmap=plt.get_cmap(cmap), vmin=vlim[1], vmax=vlim[2], shading="nearest")
    end # if
    strm = ax.streamplot(x_arr, y_arr, transpose(vx_arr), transpose(vy_arr), color=streamline_color, density=streamline_density, linewidth=streamline_lw, start_points=streamline_start_points)
    if (marker_xs != nothing) && (marker_ys != nothing)
        ax.scatter(marker_xs, marker_ys, s=marker_s, c=marker_c, marker=marker_type)
    end # if
    if cbar_label == nothing
        cbar = fig.colorbar(pmesh, ax=ax, location="right", aspect=30)
    else
        cbar = fig.colorbar(pmesh, ax=ax, location="right", aspect=30, label=cbar_label)
    end # if
    # cbar.ax.xaxis.label.set_size(12)
    cbar.ax.tick_params(which="both", length=2, direction="in")
    cbar.ax.xaxis.set_ticks_position("both")

    if fig_ax == nothing
        set_plot_defaults(fig, ax, addGrid=false)
        save_plot(file_name, save_path; saveDirectly=saveDirectly)
        plt.show()
    end # if
end # function

function plotN_us_vs_xy(x_arr, y_arr, u_arr_vec2, save_path; fig_size=nothing,
    x_name=raw"$x$", y_name=raw"$y$", x_lim_vec2=nothing, y_lim_vec2=nothing,
    ax_title_vec2=nothing, cbar_label_vec2=nothing,
    marker_xs_vec2=nothing, marker_ys_vec2=nothing, marker_s_vec2=nothing,
    marker_c_vec2=nothing, marker_type_vec2=nothing, cmap_vec2=nothing,
    vlim_vec2=nothing, saveDirectly=false, file_name="N_us_vs_xy")
    N_rows = length(u_arr_vec2)
    N_cols = length(u_arr_vec2[1])
    if fig_size == nothing
        fig_size = (9*N_rows, 6*N_cols)
    end # if
    if x_lim_vec2 == nothing
        x_lim_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if y_lim_vec2 == nothing
        y_lim_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if ax_title_vec2 == nothing
        ax_title_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if cbar_label_vec2 == nothing
        cbar_label_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_xs_vec2 == nothing
        marker_xs_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_ys_vec2 == nothing
        marker_ys_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_s_vec2 == nothing
        marker_s_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_c_vec2 == nothing
        marker_c_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_type_vec2 == nothing
        marker_type_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if cmap_vec2 == nothing
        cmap_vec2 = [["viridis" for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if vlim_vec2 == nothing
        vlim_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if

    fig, ax = plt.subplots(N_rows, N_cols)
    fig.set_size_inches(fig_size...)
    plt.subplots_adjust(left=0.15, bottom=0.12, top=0.95, wspace=0.1, hspace=0.25, right=0.97)
    for i in 1:N_rows
        for j in 1:N_cols
            u_arr = u_arr_vec2[i][j]
            x_lim = x_lim_vec2[i][j]
            y_lim = y_lim_vec2[i][j]
            ax_title = ax_title_vec2[i][j]
            cbar_label = cbar_label_vec2[i][j]
            cmap = cmap_vec2[i][j]
            vlim = vlim_vec2[i][j]
            if u_arr != nothing
                if ax_title == nothing
                    # ax_title = raw"$u$"
                    ax_title = ""
                end # if

                if N_rows != 1
                    plot_u_vs_xy(x_arr, y_arr, u_arr, save_path; x_name=x_name, y_name=y_name, x_lim=x_lim, y_lim=y_lim,
                         ax_title=ax_title, cbar_label=cbar_label, cmap=cmap, vlim=vlim, fig_ax=(fig, ax[i, j]))
                else
                    plot_u_vs_xy(x_arr, y_arr, u_arr, save_path; x_name=x_name, y_name=y_name, x_lim=x_lim, y_lim=y_lim,
                         ax_title=ax_title, cbar_label=cbar_label, cmap=cmap, vlim=vlim, fig_ax=(fig, ax[j]))
                end # if
            end # if

            marker_xs = marker_xs_vec2[i][j]
            marker_ys = marker_ys_vec2[i][j]
            marker_s = marker_s_vec2[i][j]
            marker_c = marker_c_vec2[i][j]
            marker_type = marker_type_vec2[i][j]
            if (marker_xs != nothing) && (marker_ys != nothing)
                if marker_s == nothing
                    marker_s = 5.0
                end # if
                if marker_c == nothing
                    marker_c = "w"
                end # if
                if marker_type == nothing
                    marker_type = "o"
                end # if

                ax[i, j].scatter(marker_xs, marker_ys, s=marker_s, c=marker_c, marker=marker_type)
            end # if
        end # for
    end # for

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.2)
    save_plot(file_name, save_path; saveDirectly=saveDirectly)
    plt.show()
end # function

# NOTE: Unused.
function plotN_us_w_streamlines_vs_xy(x_arr, y_arr, u_arr_vec2, vx_arr_vec2, vy_arr_vec2, save_path; x_name=raw"$x$", y_name=raw"$y$", x_lim_vec2=nothing, y_lim_vec2=nothing, ax_title_vec2=nothing,
    cbar_label_vec2=nothing, streamline_color="w", streamline_density=1, streamline_lw=2,
    streamline_start_points_vec2=nothing, marker_xs_vec2=nothing, marker_ys_vec2=nothing, marker_s_vec2=nothing,
    marker_c_vec2=nothing, marker_type_vec2=nothing, cmap_vec2=nothing, vlim_vec2=nothing, saveDirectly=true,
    file_name="N_us_w_streamlines_vs_xy")
    N_rows = length(u_arr_vec2)
    N_cols = length(u_arr_vec2[1])
    if fig_size == nothing
        fig_size = (9*N_rows, 6*N_cols)
    end # if
    if x_lim_vec2 == nothing
        x_lim_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if y_lim_vec2 == nothing
        y_lim_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if ax_title_vec2 == nothing
        ax_title_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if cbar_label_vec2 == nothing
        cbar_label_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_xs_vec2 == nothing
        marker_xs_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_ys_vec2 == nothing
        marker_ys_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_s_vec2 == nothing
        marker_s_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_c_vec2 == nothing
        marker_c_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if marker_type_vec2 == nothing
        marker_type_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if cmap_vec2 == nothing
        cmap_vec2 = [["viridis" for j in 1:N_cols] for i in 1:N_rows]
    end # if
    if vlim_vec2 == nothing
        vlim_vec2 = [[nothing for j in 1:N_cols] for i in 1:N_rows]
    end # if

    fig, ax = plt.subplots(N_rows, N_cols)
    fig.set_size_inches(fig_size...)
    plt.subplots_adjust(left=0.15, bottom=0.12, top=0.95, wspace=0.1, hspace=0.25, right=0.97)
    for i in 1:N_rows
        for j in 1:N_cols
            u_arr = u_arr_vec2[i][j]
            vx_arr = vx_arr_vec2[i][j]
            vy_arr = vy_arr_vec2[i][j]
            x_lim = x_lim_vec2[i][j]
            y_lim = y_lim_vec2[i][j]
            ax_title = ax_title_vec2[i][j]
            cbar_label = cbar_label_vec2[i][j]
            streamline_start_points = streamline_start_points_vec2[i][j]
            marker_xs = marker_xs_vec2[i][j]
            marker_ys = marker_ys_vec2[i][j]
            marker_s = marker_s_vec2[i][j]
            marker_c = marker_c_vec2[i][j]
            marker_type = marker_type_vec2[i][j]
            cmap = cmap_vec2[i][j]
            vlim = vlim_vec2[i][j]

            if u_arr != nothing
                if ax_title == nothing
                    ax_title = raw"$u$"
                end # if

                if N_rows != 1
                    plot_u_w_streamlines_vs_xy(x_arr, y_arr, u_arr, vx_arr, vy_arr, save_path; x_name=x_name, y_name=y_name, x_lim=x_lim, y_lim=y_lim,
                        ax_title=ax_title, cbar_label=cbar_label, streamline_color=streamline_color, streamline_density=streamline_density, streamline_lw=streamline_lw, streamline_start_points=streamline_start_points,
                        marker_xs=marker_xs, marker_ys=marker_ys, marker_s=marker_s, marker_c=marker_c, marker_type=marker_type, cmap=cmap,
                        vlim=vlim, saveDirectly=saveDirectly, fig_ax=(fig, ax[i, j]))
                else
                    plot_u_w_streamlines_vs_xy(x_arr, y_arr, u_arr, vx_arr, vy_arr, save_path; x_name=x_name, y_name=y_name, x_lim=x_lim, y_lim=y_lim,
                        ax_title=ax_title, cbar_label=cbar_label, streamline_color=streamline_color, streamline_density=streamline_density, streamline_lw=streamline_lw, streamline_start_points=streamline_start_points,
                        marker_xs=marker_xs, marker_ys=marker_ys, marker_s=marker_s, marker_c=marker_c, marker_type=marker_type, cmap=cmap,
                        vlim=vlim, saveDirectly=saveDirectly, fig_ax=(fig, ax[j]))
                end # if
            end # if
        end # for
    end # for

    set_plot_defaults(fig, ax, addGrid=false)
    plt.subplots_adjust(hspace=0.2)
    save_plot(file_name, save_path; saveDirectly=saveDirectly)
    plt.show()
end # function
