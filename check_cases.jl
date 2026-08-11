# Check the (suite, problem, size) triples in data/results/*.csv against cases.jl.
#
#   julia --project=. check_cases.jl [results_dir]
#
# Only suites with at least one row are checked; exits 1 if any is incomplete.
# cases.jl is parsed, not include()d, so the model packages never load.

# Refuse to run on a login node; EXA_ALLOW_LOGIN=1 overrides.
if occursin(r"^login", gethostname()) && get(ENV, "EXA_ALLOW_LOGIN", "") != "1"
    println(stderr, "check_cases: refusing to run on login node $(gethostname()). ",
            "Submit a job (mit_quicktest, 15 min cap) or set EXA_ALLOW_LOGIN=1.")
    exit(2)
end

using CSV, DataFrames

const BENCH = @__DIR__
const RESULTS = length(ARGS) >= 1 ? ARGS[1] : joinpath(BENCH, "data", "results")

casefile = get(ENV, "EXA_QUICK", "") != "" ? "cases_quick.jl" : "cases.jl"
ast = Meta.parse("begin\n" * read(joinpath(BENCH, casefile), String) * "\nend")

"Pull `const NAME = [...]` out of the parsed file."
function const_vector(ast, name::Symbol)
    for ex in ast.args
        ex isa Expr || continue
        if ex.head === :const && ex.args[1] isa Expr && ex.args[1].head === :(=) &&
           ex.args[1].args[1] === name
            return ex.args[1].args[2]
        end
    end
    error("could not find const $name")
end

"Each element is (name = \"x\", model = ..., sizes = [...]); take name and sizes."
function name_size_pairs(vec_ex)
    out = Tuple{String,String}[]
    for el in vec_ex.args
        el isa Expr || continue
        nm, sizes = nothing, nothing
        for kw in el.args
            kw isa Expr && kw.head === :(=) || continue
            kw.args[1] === :name  && (nm = kw.args[2])
            kw.args[1] === :sizes && (sizes = kw.args[2])
        end
        (nm === nothing || sizes === nothing) && continue
        for s in sizes.args
            push!(out, (String(nm), string(eval(s))))
        end
    end
    out
end

expected = Set{Tuple{String,String,String}}()
for (n, s) in name_size_pairs(const_vector(ast, :LV_CASES))
    push!(expected, ("LV", n, s))
end
for (n, s) in name_size_pairs(const_vector(ast, :COPS_CASES))
    push!(expected, ("COPS", n, s))
end
opf = String[String(x) for x in const_vector(ast, :OPF_CASES).args if x isa String]
for form in ("polar", "rect"), c in opf
    push!(expected, ("OPF-" * form, c, c))
end

# Select suite CSVs by schema, not filename: compare_*.csv has no suite/problem/size columns.
function is_suite_csv(f)
    try
        hdr = split(first(eachline(f)), ",")
        return all(c -> c in hdr, ("suite", "problem", "size"))
    catch
        return false
    end
end

allcsv = filter(f -> endswith(f, ".csv") && !startswith(basename(f), "partial_") &&
                     basename(f) != "combined.csv",
                joinpath.(RESULTS, readdir(RESULTS)))
csvs = filter(is_suite_csv, allcsv)
skipped = length(allcsv) - length(csvs)
skipped > 0 && println("check_cases: skipped $skipped non-suite csv(s) (e.g. section 8.4 compare output)")
if isempty(csvs)
    println("check_cases: no suite CSVs in $RESULTS — nothing to check (this is normal for a compare-only leg)")
    exit(0)
end
if isempty(allcsv)
    println("check_cases: no non-partial CSVs in $RESULTS")
    exit(1)
end

present = Set{Tuple{String,String,String}}()
for f in csvs
    df = CSV.read(f, DataFrame; types = Dict(:suite => String, :problem => String, :size => String))
    for r in eachrow(df)
        push!(present, (r.suite, r.problem, r.size))
    end
end

suites = sort(unique(s for (s, _, _) in present))
println("check_cases: ", length(csvs), " csv(s), suites present: ", join(suites, ", "))

rc = Ref(0)
for suite in suites
    exp_s = sort([(p, z) for (s, p, z) in expected if s == suite])
    got_s = Set((p, z) for (s, p, z) in present if s == suite)
    missing_s = [x for x in exp_s if !(x in got_s)]
    extra_s = sort([x for x in got_s if !(x in Set(exp_s))])
    status = isempty(missing_s) && isempty(extra_s) ? "OK" : "INCOMPLETE"
    println("  $suite: $(length(got_s))/$(length(exp_s)) cases  [$status]")
    for (p, z) in missing_s
        println("      MISSING  $suite  $p/$z")
        rc[] = 1
    end
    for (p, z) in extra_s
        println("      UNEXPECTED  $suite  $p/$z")
        rc[] = 1
    end
end
exit(rc[])
