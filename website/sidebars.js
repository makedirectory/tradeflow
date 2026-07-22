// @ts-check
// Two sidebars: a task-oriented Usage guide and an Engineering wiki.

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  usage: [
    'intro',
    {
      type: 'category',
      label: 'Usage',
      collapsed: false,
      items: [
        'usage/installation',
        'usage/configuration',
        'usage/brokers',
        'usage/scanning',
        'usage/backtesting',
        'usage/live-trading',
        'usage/optimization',
        'usage/portfolio',
        'usage/alphas',
        'usage/risk',
        'usage/information',
        'usage/horizon',
        'usage/walk-forward',
        'usage/agents',
      ],
    },
  ],
  engineering: [
    {
      type: 'category',
      label: 'Engineering Wiki',
      collapsed: false,
      items: [
        'engineering/architecture',
        'engineering/philosophy',
        'engineering/separation-of-concerns',
        'engineering/broker-abstraction',
        'engineering/data-flow',
        'engineering/data-panel',
        'engineering/strategies',
        'engineering/scanners',
        'engineering/indicators',
        'engineering/engine',
        'engineering/optimization',
        'engineering/portfolio',
        'engineering/portfolio-construction',
        'engineering/multi-period-trading',
        'engineering/alphas',
        'engineering/multi-signal',
        'engineering/risk-model',
        'engineering/transaction-costs',
        'engineering/information-analysis',
        'engineering/attribution',
        'engineering/information-horizon',
        'engineering/evaluation-metrics',
        'engineering/walk-forward',
        'engineering/mcp-server',
        'engineering/research-agent',
        'engineering/testing',
        'engineering/coding-standards',
        'engineering/extending',
      ],
    },
  ],
};

module.exports = sidebars;
