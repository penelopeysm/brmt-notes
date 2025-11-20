```julia

using RDatasets: dataset, Not

# Load the epilepsy dataset and select only the rows where Period is "1"
full_df = dataset("HSAUR", "epilepsy")
df = full_df[full_df.Period .== "1", Not("Period")]
df

# Define the formula
using StatsModels
f = @formula(SeizureRate ~ 1 + Treatment)

# Get the data
schema(f, df)
a = apply_schema(f, schema(f, df))
resp, pred = modelcols(a, df)

using Turing
struct TuringPredictor{T<:StatsModels.AbstractTerm,D<:Distribution}
    term::T
    param_name::Symbol
    prior::D
end
Base.show(io::IO, tp::TuringPredictor) = print(io, "$(tp.param_name) ~ $(tp.prior)")
struct TuringModelSpec{V<:Vector{<:TuringPredictor}}
    # directly passed in from the formula
    likelihood_term::Term
    # must take a single argument (the linear combination of predictors)
    # and return a distribution
    link_function::Function
    predictors::V
end
function Base.show(io::IO, tms::TuringModelSpec)
    println(io, "Likelihood:\n  $(tms.likelihood_term) ~ $(tms.link_function)(µ)")
    println(io, "Predictors:")
    for p in tms.predictors
        println(io, "  ", p)
    end
end

make_param_name(t::ConstantTerm) = :Intercept
make_param_name(t::Term) = Symbol("b_$(t.sym)")
function TuringModelSpec(
    f::StatsModels.FormulaTerm,
    link_function::Function,
    priors::NamedTuple=NamedTuple()
)
    # f.lhs contains the response
    likelihood_term = f.lhs
    # associate each predictor with its prior
    predictors = map(f.rhs) do term
        param_name = make_param_name(term)
        prior = get(priors, param_name, Flat())
        TuringPredictor(term, param_name, prior)
    end
    TuringModelSpec(likelihood_term, link_function, collect(predictors))
end
poisson_link(µ) = Poisson(exp(µ))
TuringModelSpec(f, poisson_link, (; b_Treatment = Normal(0, 10)))







```
