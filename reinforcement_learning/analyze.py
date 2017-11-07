from pylab import *



# TODO: need to plot ALL query_fns together... (pain in the ass!!)


# FIGURE 1: rendering


# FIGURE 2: MSE of reward predictions
figure()
[plot(np.mean(rer, axis=0), label=lab) for rer, lab in zip(reward_err_experience, query_fns)]
legend()
xlabel('episode')
ylabel('MSE of reward prediction')

figure()
[plot(np.mean(rer, axis=0), label=lab) for rer, lab in zip(reward_err_uniform, query_fns)]
legend()
xlabel('episode')
ylabel('MSE of reward prediction')




# FIGURE 3: performance
# TODO: cum_sum cum_queries (etc.)
# TODO: labels

cum_rewards = np.cumsum(rewards_true.reshape((len(query_fns), num_seeds, -1)), axis=-1)
cum_queries = np.cumsum(queries.reshape((len(query_fns), num_seeds, -1)), axis=-1)

figure()
for qfn, query_fn in enumerate(query_fns):
    cr = cum_rewards[qfn]
    cq = cum_queries[qfn]
    # 
    subplot(2,2,1)
    ylabel('rewards')
    plot(cum_rewards[qfn].mean(0))
    # 
    subplot(2,2,3)
    xlabel('step')
    ylabel('#queries')
    plot(cum_queries[qfn].mean(0))
    # 
    subplot(2,2,2)
    ylabel('performance (c=.1)')
    plot((cr - .1 * cq).mean(0))# for r,nq in zip(cum_rewards[qfn], cum_queries[qfn])])
    # 
    subplot(2,2,4)
    xlabel('step')
    ylabel('performance (c=1)')
    plot((cr - 1 * cq).mean(0), label=query_fn)# for r,nq in zip(cum_rewards[qfn], cum_queries[qfn])])
    #plot([(r - nq).mean(0) for r,nq in zip(cum_rewards[qfn], cum_queries[qfn])], label=query_fn)
legend()



