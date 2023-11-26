using CSV, DataFrames
using StatsBase

df_renal = CSV.read("../data/IKEM - renal - training - SEX.csv", DataFrame)

df_res1 = CSV.read("../data/res1.txt", DataFrame)
df_res2 = CSV.read("../data/res2.txt", DataFrame)
df_res3 = CSV.read("../data/res3.txt", DataFrame)
df_res4 = CSV.read("../data/res4.txt", DataFrame)
df_res5 = CSV.read("../data/res5.txt", DataFrame)

df_diag = CSV.read("../data/IKEM - Dg - training.csv", DataFrame)



function combine_renal_diag(df_renal, df_diag)
    dfr = select(df_renal, Not([:ID]))
    dfr = select(df_renal, [:patient, :age, :BMI, :])
    re = r"([A-Z]{1}[0-9]{2})(\.[0-9]{0,2})?"
    dfd = combine(df_diag, :,  :DgCode => ByRow(x->(m=match(re, x); typeof(m)<:RegexMatch ? m.match : nothing)) => :Dg)
    unique_dgs = unique(dfd[!, :Dg])[2:end]  # nothing is first

    old_new = [d=>"d_$d" for d in unique_dgs]
    dfd[:, :Dg] = replace(dfd[:, :Dg], old_new...)

    ps = df_renal[:, :patient]

    counts = dfd[:, :Dg] |> countmap |> collect
    counts_sorted = sort(counts, by=x->x.second)
    unique_dgs = unique(dfd[!, :Dg]) 
    MIN_PATIENTS = 20
    to_keep = map(x->x.first => x.second > MIN_PATIENTS, counts_sorted)
    to_keep = Dict(to_keep)
    to_keep[nothing] = false
    ids = [to_keep[d] ? true : false for d in unique_dgs] 
    new_unique_dgs = unique_dgs[ids]

    df_patient_vs_diag = DataFrame("patient"=>ps, [dg=>zeros(Bool, length(ps)) for dg in new_unique_dgs]...)

    for r in eachrow(dfd)
        p = r.patient
        d = r.Dg
        if to_keep[d]
            df_patient_vs_diag[df_patient_vs_diag[!, :patient] .== p, d] .= true
        end
    end

    rename!(dfr, :age=>:x_age, :Sex =>:x_sex, :BMI=>:x_BMI, :KREA=>:y_KREA, :ACR=>:y_ACR)
    df_final = innerjoin(dfr, df_patient_vs_diag, on=:patient)
    CSV.write("honza_jirka_martin_data.csv", df_final)

end
