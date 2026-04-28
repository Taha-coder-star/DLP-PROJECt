// All training results hardcoded from models/*.json and README.md

export const DATASET = {
  totalRows: 326985,
  users: 1000,
  insiders: 70,
  trainRows: 261220,
  testRows: 65765,
  trainStart: '2010-01-02',
  trainEnd: '2011-02-05',
  testStart: '2011-02-06',
  testEnd: '2011-05-16',
  features: 32,
  certRelease: '4.2',
};

export const FEATURE_CATEGORIES = [
  { category: 'Email Activity', count: 16, color: '#3b82f6',
    features: ['email_count','total_size','avg_size','attachment_count','emails_with_attachments',
                'attachment_ratio','after_hours_ratio','bcc_ratio','cc_ratio','bcc_count',
                'cc_count','unique_pcs','avg_recipients','max_recipients','avg_content_words','max_content_words'] },
  { category: 'Logon / Session', count: 4, color: '#8b5cf6',
    features: ['logon_count','logoff_count','after_hours_logons','unique_logon_pcs'] },
  { category: 'USB / Removable', count: 2, color: '#f59e0b',
    features: ['usb_connect_count','usb_disconnect_count'] },
  { category: 'File System', count: 5, color: '#ef4444',
    features: ['file_total','file_to_removable','file_from_removable','file_write_count','file_after_hours'] },
  { category: 'Psychometric (OCEAN)', count: 5, color: '#10b981',
    features: ['openness','conscientiousness','extraversion','agreeableness','neuroticism'] },
];

export const ISOLATION_FOREST = {
  params: {
    nEstimators: 300,
    contamination: 0.03,
    inputFeatures: 32,
    suspiciousThreshold: 0.496,
    highThreshold: 0.642,
  },
  day: {
    rocAuc: 0.3819,
    avgPrecision: 0.0029,
    precision: 0.001,
    recall: 0.0124,
    f1: 0.0019,
    tp: 3, fp: 2868, fn: 239, tn: 57727,
    nInsiders: 242, nTotal: 60837,
  },
  user: {
    rocAuc: 0.4624,
    avgPrecision: 0.0627,
    precision: 0.0588,
    recall: 0.4571,
    f1: 0.1042,
    tp: 32, fp: 512, fn: 38, tn: 418,
    nInsiders: 70, nTotal: 1000,
  },
  suspiciousRows: 12903,
  highRows: 3276,
  topAnomalies: [
    { user: 'KBP0008', date: '2010-06-10', score: 1.000 },
    { user: 'KBP0008', date: '2010-03-03', score: 0.977 },
    { user: 'KBP0008', date: '2010-06-21', score: 0.971 },
    { user: 'KBP0008', date: '2010-11-12', score: 0.963 },
    { user: 'KBP0008', date: '2010-11-16', score: 0.961 },
    { user: 'KBP0008', date: '2010-06-03', score: 0.958 },
    { user: 'TVS0006', date: '2010-05-11', score: 0.942 },
    { user: 'TVS0006', date: '2010-01-05', score: 0.941 },
    { user: 'KBP0008', date: '2010-12-28', score: 0.935 },
    { user: 'KBP0008', date: '2010-06-22', score: 0.931 },
  ],
};

export const LSTM = {
  params: {
    windowSize: 7,
    hiddenDim: 32,
    latentDim: 16,
    epochs: 20,
    batchSize: 256,
    lr: 0.001,
    loss: 'Mean Squared Reconstruction Error',
    optimizer: 'Adam',
    suspiciousThreshold: 0.7228,
    highThreshold: 0.9288,
  },
  day: {
    rocAuc: 0.6032,
    avgPrecision: 0.0525,
    precision: 0.0645,
    recall: 0.2353,
    f1: 0.1013,
    tp: 296, fp: 4291, fn: 962, tn: 60216,
    nInsiders: 1258, nTotal: 65765,
  },
  user: {
    rocAuc: 0.5618,
    avgPrecision: 0.0917,
    precision: 0.07,
    recall: 1.0,
    f1: 0.1308,
    tp: 70, fp: 930, fn: 0, tn: 0,
    nInsiders: 70, nTotal: 1000,
  },
  suspiciousRows: 12936,
  highRows: 4945,
  undeterminedRows: 6000,
  topAnomalies: [
    { user: 'BER0314', date: '2011-03-22', score: 1.0, severity: 'high' },
    { user: 'DKB0631', date: '2011-03-28', score: 1.0, severity: 'high' },
    { user: 'DKB0631', date: '2011-03-29', score: 1.0, severity: 'high' },
    { user: 'DKB0631', date: '2011-03-30', score: 1.0, severity: 'high' },
    { user: 'DKB0631', date: '2011-03-31', score: 1.0, severity: 'high' },
    { user: 'WLV0566', date: '2011-04-18', score: 1.0, severity: 'high' },
    { user: 'WLV0566', date: '2011-04-19', score: 1.0, severity: 'high' },
    { user: 'WLV0566', date: '2011-04-20', score: 1.0, severity: 'high' },
    { user: 'MIM0712', date: '2011-04-13', score: 1.0, severity: 'high' },
    { user: 'BER0314', date: '2011-04-15', score: 1.0, severity: 'high' },
    { user: 'SJC0333', date: '2011-03-04', score: 1.0, severity: 'high' },
    { user: 'SJC0333', date: '2011-03-03', score: 1.0, severity: 'high' },
    { user: 'AMJ0297', date: '2011-04-07', score: 1.0, severity: 'high' },
    { user: 'AMJ0297', date: '2011-04-05', score: 1.0, severity: 'high' },
    { user: 'SHR0334', date: '2011-04-18', score: 1.0, severity: 'high' },
    { user: 'SHR0334', date: '2011-04-15', score: 1.0, severity: 'high' },
    { user: 'SHR0334', date: '2011-04-14', score: 1.0, severity: 'high' },
    { user: 'SHR0334', date: '2011-04-13', score: 1.0, severity: 'high' },
    { user: 'DLH0029', date: '2011-04-19', score: 1.0, severity: 'high' },
    { user: 'DLH0029', date: '2011-04-18', score: 1.0, severity: 'high' },
  ],
};

export const DLP_TIERS = [
  {
    label: 'RESTRICTED', score: 3, color: '#ef4444',
    signals: 'PII (SSN, passport, routing numbers), credentials, PE executables to USB, medical records',
  },
  {
    label: 'SENSITIVE', score: 2, color: '#f59e0b',
    signals: 'Salary/payroll, M&A/strategy docs, confidential/NDA, office docs to USB, PDF/archives',
  },
  {
    label: 'INTERNAL', score: 1, color: '#3b82f6',
    signals: 'General internal documents, any removable-media read activity',
  },
  {
    label: 'PUBLIC', score: 0, color: '#6b7280',
    signals: 'No indicators triggered — standard communications',
  },
];

export const RISK_SIGNALS = [
  { key: 'lstm_p95',            label: 'LSTM P95',            defaultWeight: 0.45,  gaWeight: 0.3036, gaThreshold: 0.3373, defaultThreshold: 0.70, color: '#10b981', desc: 'LSTM reconstruction-error 95th percentile' },
  { key: 'after_hours',         label: 'After Hours',         defaultWeight: 0.13,  gaWeight: 0.0300, gaThreshold: 0.95,   defaultThreshold: 0.50, color: '#3b82f6', desc: 'Mean after-hours email / logon fraction' },
  { key: 'bcc_usage',           label: 'BCC Usage',           defaultWeight: 0.09,  gaWeight: 0.0391, gaThreshold: 0.5913, defaultThreshold: 0.50, color: '#f59e0b', desc: 'Mean BCC email ratio' },
  { key: 'file_exfil',          label: 'File Exfil',          defaultWeight: 0.09,  gaWeight: 0.2980, gaThreshold: 0.7094, defaultThreshold: 0.50, color: '#ef4444', desc: 'Files written to removable media (rate)' },
  { key: 'usb_activity',        label: 'USB Activity',        defaultWeight: 0.09,  gaWeight: 0.2817, gaThreshold: 0.482,  defaultThreshold: 0.50, color: '#8b5cf6', desc: 'Total USB connect events' },
  { key: 'multi_pc',            label: 'Multi PC',            defaultWeight: 0.05,  gaWeight: 0.0476, gaThreshold: 0.3314, defaultThreshold: 0.50, color: '#ec4899', desc: 'Max distinct workstations accessed' },
  { key: 'content_sensitivity', label: 'DLP Sensitivity',     defaultWeight: 0.10,  gaWeight: 0.10,   gaThreshold: 0.50,   defaultThreshold: 0.50, color: '#14b8a6', desc: 'Mean daily max DLP sensitivity score' },
];

export const EVALUATION_RESULTS = [
  { config: 'LSTM + GA Weights',     agg: 'score_p95', k: 50, precision: 0.32,  recall: 0.2286, f1: 0.2667, tp: 16, best: true },
  { config: 'LSTM + Domain Weights', agg: 'score_p95', k: 50, precision: 0.30,  recall: 0.2143, f1: 0.2500, tp: 15, best: false },
  { config: 'LSTM + GA Weights',     agg: 'score_p95', k: 20, precision: 0.50,  recall: 0.1400, f1: 0.2222, tp: 10, best: false },
  { config: 'Isolation Forest',      agg: 'any',       k: 50, precision: null,  recall: null,   f1: null,   tp: null, best: false },
];

export const GA = {
  params: {
    popSize: 60, maxGens: 100, tournamentK: 5,
    crossoverProb: 0.5, mutationSigma: 0.04, mutationRate: 0.25,
    elitismN: 4, patience: 25, fitnessAlpha: 0.8, fitnessBeta: 0.2, topK: 50,
  },
  convergedAt: 26,
  bestFitness: 0.41336,
  metrics: { k: 50, tp: 16, fp: 34, fn: 54, precision: 0.32, recall: 0.2286, f1: 0.2667, flagCoverage: 1.0 },
  baselineMetrics: { k: 50, tp: 15, precision: 0.30, recall: 0.2143, f1: 0.25 },
  improvement: { deltaF1: 0.0167, deltaPrecision: 0.02, deltaRecall: 0.0143 },
  convergenceHistory: [
    { gen: 1,  best: 0.413333, mean: 0.319133 },
    { gen: 2,  best: 0.413333, mean: 0.379111 },
    { gen: 3,  best: 0.413333, mean: 0.402000 },
    { gen: 4,  best: 0.413333, mean: 0.407556 },
    { gen: 5,  best: 0.413333, mean: 0.409111 },
    { gen: 6,  best: 0.413333, mean: 0.407511 },
    { gen: 7,  best: 0.413333, mean: 0.409111 },
    { gen: 8,  best: 0.413333, mean: 0.409333 },
    { gen: 9,  best: 0.413333, mean: 0.407333 },
    { gen: 10, best: 0.413333, mean: 0.407556 },
    { gen: 11, best: 0.413333, mean: 0.408667 },
    { gen: 12, best: 0.413333, mean: 0.409111 },
    { gen: 13, best: 0.413333, mean: 0.408889 },
    { gen: 14, best: 0.413333, mean: 0.409044 },
    { gen: 15, best: 0.413333, mean: 0.409333 },
    { gen: 16, best: 0.413333, mean: 0.408222 },
    { gen: 17, best: 0.413333, mean: 0.411111 },
    { gen: 18, best: 0.413333, mean: 0.410444 },
    { gen: 19, best: 0.413333, mean: 0.410000 },
    { gen: 20, best: 0.413333, mean: 0.410444 },
    { gen: 21, best: 0.413333, mean: 0.410889 },
    { gen: 22, best: 0.413333, mean: 0.410667 },
    { gen: 23, best: 0.413333, mean: 0.409556 },
    { gen: 24, best: 0.413333, mean: 0.409333 },
    { gen: 25, best: 0.413333, mean: 0.409778 },
    { gen: 26, best: 0.413333, mean: 0.409333 },
  ],
};

// K-sweep P/R/F1 data (interpolated from known points)
export const K_SWEEP = [
  { k: 10,  precision: 0.50, recall: 0.071, f1: 0.125 },
  { k: 20,  precision: 0.50, recall: 0.143, f1: 0.222 },
  { k: 30,  precision: 0.40, recall: 0.171, f1: 0.240 },
  { k: 50,  precision: 0.32, recall: 0.229, f1: 0.267 },
  { k: 75,  precision: 0.21, recall: 0.229, f1: 0.219 },
  { k: 100, precision: 0.16, recall: 0.229, f1: 0.188 },
];
