using JuMP
using Test
using Ipopt
using HiGHS
using SmartInverterDOPF

@testset "Volt-VAr encoding equivalence" begin
    curve = ieee1547_curve()
    qbar = Dict(1 => 2.0)
    voltages = [0.85, 0.88, 0.89, 0.90, 0.93, 0.97, 0.985, 1.00, 1.01, 1.02, 1.06, 1.10, 1.15]

    @testset "reference curve" begin
        for v in voltages
            @test isfinite(droop_q(curve, v, 2.0))
            @test -2.0 - 1e-12 <= droop_q(curve, v, 2.0) <= 2.0 + 1e-12
        end
    end

    @testset "Big-M and Lambda agree" begin
        for vtarget in voltages
            models = Dict{Symbol,Model}()
            qvars = Dict{Symbol,Any}()
            vvars = Dict{Symbol,Any}()

            for (method, optimizer) in ((:bigm, HiGHS.Optimizer), (:lambda, HiGHS.Optimizer))
                model = Model(optimizer)
                set_silent(model)
                @variable(model, 0.80 <= v[1, 1, 1] <= 1.20)
                @variable(model, -2.0 <= q[1, 1, 1] <= 2.0)
                add_droop!(model, method, curve, v, q, 1:1, 1:1, 1:1, qbar)
                @constraint(model, v[1, 1, 1] == vtarget)
                @objective(model, Min, q[1, 1, 1]^2)
                optimize!(model)
                @test termination_status(model) in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
                models[method] = model
                qvars[method] = q
                vvars[method] = v
            end

            q_big_m = value(qvars[:bigm][1, 1, 1])
            q_lambda = value(qvars[:lambda][1, 1, 1])
            q_ref = droop_q(curve, vtarget, 2.0)

            @test q_big_m ≈ q_ref atol = 1e-7
            @test q_lambda ≈ q_ref atol = 1e-7
            @test q_big_m ≈ q_lambda atol = 1e-7
        end
    end

    @testset "Heaviside agrees with scalar reference" begin
        for vtarget in voltages
            model = Model(Ipopt.Optimizer)
            set_silent(model)
            @variable(model, 0.80 <= v[1, 1, 1] <= 1.20, start = vtarget)
            @variable(model, -2.0 <= q[1, 1, 1] <= 2.0, start = droop_q(curve, vtarget, 2.0))
            add_droop!(model, :heaviside, curve, v, q, 1:1, 1:1, 1:1, qbar)
            @constraint(model, v[1, 1, 1] == vtarget)
            @objective(model, Min, (q[1, 1, 1] - droop_q(curve, vtarget, 2.0))^2)
            optimize!(model)
            @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL)
            @test value(q[1, 1, 1]) ≈ droop_q(curve, vtarget, 2.0) atol = 1e-6
        end
    end

    @testset "custom curve remains self-consistent" begin
        custom = DroopCurve([0.90, 0.92, 0.98, 1.00, 1.03, 1.08],
                            [1.0, 1.0, 0.25, 0.0, -0.75, -1.0])
        for (v, expected) in ((0.89, 2.0), (0.91, 2.0), (0.95, 1.5),
                              (0.99, 0.125), (1.015, -0.375), (1.10, -2.0))
            @test droop_q(custom, v, 2.0) ≈ expected atol = 1e-10
        end
    end
end
