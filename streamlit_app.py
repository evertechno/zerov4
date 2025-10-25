-- ============================================
-- Invsion Connect - Supabase Database Setup
-- ============================================
-- Run this script in your Supabase SQL Editor
-- This creates all tables with Row Level Security enabled

-- ============================================
-- 1. PORTFOLIOS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS portfolios (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    portfolio_name TEXT NOT NULL,
    holdings_data JSONB NOT NULL,
    total_value NUMERIC(20, 2),
    holdings_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX IF NOT EXISTS idx_portfolios_created_at ON portfolios(created_at DESC);

-- Enable Row Level Security
ALTER TABLE portfolios ENABLE ROW LEVEL SECURITY;

-- RLS Policies for portfolios
CREATE POLICY "Users can view their own portfolios"
    ON portfolios FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own portfolios"
    ON portfolios FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own portfolios"
    ON portfolios FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own portfolios"
    ON portfolios FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================
-- 2. COMPLIANCE CONFIGS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS compliance_configs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    single_stock_limit NUMERIC(5, 2),
    single_sector_limit NUMERIC(5, 2),
    top_10_limit NUMERIC(5, 2),
    min_holdings INTEGER,
    unrated_limit NUMERIC(5, 2),
    custom_rules TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_compliance_configs_user_id ON compliance_configs(user_id);
CREATE INDEX IF NOT EXISTS idx_compliance_configs_portfolio_id ON compliance_configs(portfolio_id);

-- Enable Row Level Security
ALTER TABLE compliance_configs ENABLE ROW LEVEL SECURITY;

-- RLS Policies for compliance_configs
CREATE POLICY "Users can view their own compliance configs"
    ON compliance_configs FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own compliance configs"
    ON compliance_configs FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own compliance configs"
    ON compliance_configs FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own compliance configs"
    ON compliance_configs FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================
-- 3. ANALYSIS RESULTS TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS analysis_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    config_id UUID NOT NULL REFERENCES compliance_configs(id) ON DELETE CASCADE,
    compliance_results JSONB,
    security_compliance JSONB,
    breach_alerts JSONB,
    advanced_metrics JSONB,
    analysis_date TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_analysis_results_user_id ON analysis_results(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_portfolio_id ON analysis_results(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_config_id ON analysis_results(config_id);
CREATE INDEX IF NOT EXISTS idx_analysis_results_analysis_date ON analysis_results(analysis_date DESC);

-- Enable Row Level Security
ALTER TABLE analysis_results ENABLE ROW LEVEL SECURITY;

-- RLS Policies for analysis_results
CREATE POLICY "Users can view their own analysis results"
    ON analysis_results FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own analysis results"
    ON analysis_results FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own analysis results"
    ON analysis_results FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own analysis results"
    ON analysis_results FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================
-- 4. AI ANALYSES TABLE
-- ============================================
CREATE TABLE IF NOT EXISTS ai_analyses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    analysis_text TEXT NOT NULL,
    document_names JSONB,
    analysis_config JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ai_analyses_user_id ON ai_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_analyses_portfolio_id ON ai_analyses(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_ai_analyses_created_at ON ai_analyses(created_at DESC);

-- Enable Row Level Security
ALTER TABLE ai_analyses ENABLE ROW LEVEL SECURITY;

-- RLS Policies for ai_analyses
CREATE POLICY "Users can view their own AI analyses"
    ON ai_analyses FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own AI analyses"
    ON ai_analyses FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own AI analyses"
    ON ai_analyses FOR UPDATE
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own AI analyses"
    ON ai_analyses FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================
-- 5. USER PROFILES TABLE (Optional - for extended user info)
-- ============================================
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    full_name TEXT,
    organization TEXT,
    role TEXT,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index
CREATE INDEX IF NOT EXISTS idx_user_profiles_id ON user_profiles(id);

-- Enable Row Level Security
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- RLS Policies for user_profiles
CREATE POLICY "Users can view their own profile"
    ON user_profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can insert their own profile"
    ON user_profiles FOR INSERT
    WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
    ON user_profiles FOR UPDATE
    USING (auth.uid() = id)
    WITH CHECK (auth.uid() = id);

-- ============================================
-- 6. TRIGGERS FOR UPDATED_AT TIMESTAMPS
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Trigger for portfolios
CREATE TRIGGER update_portfolios_updated_at
    BEFORE UPDATE ON portfolios
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for compliance_configs
CREATE TRIGGER update_compliance_configs_updated_at
    BEFORE UPDATE ON compliance_configs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for user_profiles
CREATE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 7. USEFUL VIEWS (Optional but recommended)
-- ============================================

-- View: Latest portfolio analysis per portfolio
CREATE OR REPLACE VIEW latest_portfolio_analyses AS
SELECT DISTINCT ON (portfolio_id)
    ar.id,
    ar.user_id,
    ar.portfolio_id,
    p.portfolio_name,
    p.total_value,
    p.holdings_count,
    ar.analysis_date,
    ar.breach_alerts,
    cc.single_stock_limit,
    cc.single_sector_limit
FROM analysis_results ar
JOIN portfolios p ON ar.portfolio_id = p.id
JOIN compliance_configs cc ON ar.config_id = cc.id
ORDER BY portfolio_id, ar.analysis_date DESC;

-- View: Portfolio summary with analysis count
CREATE OR REPLACE VIEW portfolio_summary AS
SELECT 
    p.id,
    p.user_id,
    p.portfolio_name,
    p.total_value,
    p.holdings_count,
    p.created_at,
    COUNT(DISTINCT ar.id) as analysis_count,
    COUNT(DISTINCT ai.id) as ai_analysis_count,
    MAX(ar.analysis_date) as last_analysis_date
FROM portfolios p
LEFT JOIN analysis_results ar ON p.id = ar.portfolio_id
LEFT JOIN ai_analyses ai ON p.id = ai.portfolio_id
GROUP BY p.id, p.user_id, p.portfolio_name, p.total_value, p.holdings_count, p.created_at;

-- ============================================
-- 8. STORAGE BUCKETS (For document uploads - if needed)
-- ============================================

-- Create a storage bucket for user documents
-- Note: Run this in Supabase Dashboard > Storage, or via API
-- This is just documentation of what to create:

/*
Bucket Name: user-documents
Public: false
File size limit: 50MB
Allowed MIME types: application/pdf, text/plain

RLS Policies:
1. "Users can upload their own documents"
   - Operation: INSERT
   - Policy: bucket_id = 'user-documents' AND auth.uid()::text = (storage.foldername(name))[1]

2. "Users can view their own documents"
   - Operation: SELECT
   - Policy: bucket_id = 'user-documents' AND auth.uid()::text = (storage.foldername(name))[1]

3. "Users can delete their own documents"
   - Operation: DELETE
   - Policy: bucket_id = 'user-documents' AND auth.uid()::text = (storage.foldername(name))[1]
*/

-- ============================================
-- 9. HELPER FUNCTIONS
-- ============================================

-- Function to get user's portfolio count
CREATE OR REPLACE FUNCTION get_user_portfolio_count(user_uuid UUID)
RETURNS INTEGER AS $
BEGIN
    RETURN (SELECT COUNT(*) FROM portfolios WHERE user_id = user_uuid);
END;
$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get user's total portfolio value
CREATE OR REPLACE FUNCTION get_user_total_portfolio_value(user_uuid UUID)
RETURNS NUMERIC AS $
BEGIN
    RETURN (SELECT COALESCE(SUM(total_value), 0) FROM portfolios WHERE user_id = user_uuid);
END;
$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check if portfolio has breaches
CREATE OR REPLACE FUNCTION portfolio_has_breaches(portfolio_uuid UUID)
RETURNS BOOLEAN AS $
DECLARE
    latest_analysis JSONB;
BEGIN
    SELECT breach_alerts INTO latest_analysis
    FROM analysis_results
    WHERE portfolio_id = portfolio_uuid
    ORDER BY analysis_date DESC
    LIMIT 1;
    
    RETURN jsonb_array_length(latest_analysis) > 0;
END;
$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- 10. INITIAL DATA & CONSTRAINTS
-- ============================================

-- Add check constraints for validation
ALTER TABLE compliance_configs
    ADD CONSTRAINT check_single_stock_limit CHECK (single_stock_limit >= 0 AND single_stock_limit <= 100),
    ADD CONSTRAINT check_single_sector_limit CHECK (single_sector_limit >= 0 AND single_sector_limit <= 100),
    ADD CONSTRAINT check_top_10_limit CHECK (top_10_limit >= 0 AND top_10_limit <= 100),
    ADD CONSTRAINT check_unrated_limit CHECK (unrated_limit >= 0 AND unrated_limit <= 100),
    ADD CONSTRAINT check_min_holdings CHECK (min_holdings >= 0);

ALTER TABLE portfolios
    ADD CONSTRAINT check_total_value CHECK (total_value >= 0),
    ADD CONSTRAINT check_holdings_count CHECK (holdings_count >= 0);

-- ============================================
-- 11. AUDIT LOG TABLE (Optional - for compliance tracking)
-- ============================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    action TEXT NOT NULL,
    table_name TEXT NOT NULL,
    record_id UUID,
    old_data JSONB,
    new_data JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_table_name ON audit_logs(table_name);

-- Enable Row Level Security
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- RLS Policies for audit_logs (only read access for users)
CREATE POLICY "Users can view their own audit logs"
    ON audit_logs FOR SELECT
    USING (auth.uid() = user_id);

-- ============================================
-- 12. REAL-TIME SUBSCRIPTIONS (Enable for tables)
-- ============================================

-- Enable real-time for portfolios
ALTER PUBLICATION supabase_realtime ADD TABLE portfolios;

-- Enable real-time for analysis_results
ALTER PUBLICATION supabase_realtime ADD TABLE analysis_results;

-- ============================================
-- 13. DATABASE STATISTICS VIEWS
-- ============================================

-- View: User statistics
CREATE OR REPLACE VIEW user_statistics AS
SELECT 
    u.id as user_id,
    u.email,
    up.full_name,
    COUNT(DISTINCT p.id) as total_portfolios,
    COALESCE(SUM(p.total_value), 0) as total_portfolio_value,
    COUNT(DISTINCT ar.id) as total_analyses,
    COUNT(DISTINCT ai.id) as total_ai_analyses,
    MAX(p.created_at) as last_portfolio_created,
    MAX(ar.analysis_date) as last_analysis_date
FROM auth.users u
LEFT JOIN user_profiles up ON u.id = up.id
LEFT JOIN portfolios p ON u.id = p.user_id
LEFT JOIN analysis_results ar ON u.id = ar.user_id
LEFT JOIN ai_analyses ai ON u.id = ai.user_id
GROUP BY u.id, u.email, up.full_name;

-- ============================================
-- 14. CLEANUP FUNCTIONS
-- ============================================

-- Function to delete old analysis results (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_analyses(days_to_keep INTEGER DEFAULT 365)
RETURNS INTEGER AS $
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM analysis_results
    WHERE analysis_date < NOW() - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to delete old AI analyses (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_ai_analyses(days_to_keep INTEGER DEFAULT 180)
RETURNS INTEGER AS $
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM ai_analyses
    WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$ LANGUAGE plpgsql SECURITY DEFINER;

-- ============================================
-- 15. GRANTS (if using service role)
-- ============================================

-- Grant necessary permissions to authenticated users
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO authenticated;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO authenticated;

-- ============================================
-- SETUP COMPLETE
-- ============================================

-- Verify tables were created
SELECT 
    table_name,
    (SELECT COUNT(*) FROM information_schema.table_constraints 
     WHERE constraint_type = 'PRIMARY KEY' 
     AND table_name = tables.table_name) as has_primary_key,
    (SELECT COUNT(*) FROM pg_policies 
     WHERE tablename = tables.table_name) as rls_policies_count
FROM information_schema.tables
WHERE table_schema = 'public' 
AND table_type = 'BASE TABLE'
AND table_name IN ('portfolios', 'compliance_configs', 'analysis_results', 'ai_analyses', 'user_profiles', 'audit_logs')
ORDER BY table_name;

-- ============================================
-- NOTES FOR DEPLOYMENT
-- ============================================

/*
1. Run this entire script in Supabase SQL Editor
2. Verify all tables have RLS enabled: 
   SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname = 'public';
3. Test authentication and policies with a test user
4. Configure email templates in Supabase Dashboard > Authentication > Email Templates
5. Set up rate limiting in Supabase Dashboard > Settings > API
6. Enable email confirmations: Dashboard > Authentication > Settings > Enable email confirmations
7. Configure SMTP for production (optional): Dashboard > Settings > Auth > SMTP Settings

SECRETS.TOML CONFIGURATION:
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-public-key"

[kite]
api_key = "your-kite-api-key"
api_secret = "your-kite-api-secret"
redirect_uri = "your-redirect-uri"

[google_gemini]
api_key = "your-gemini-api-key"

REQUIRED PYTHON PACKAGES:
pip install streamlit pandas plotly numpy ta google-generativeai kiteconnect supabase PyMuPDF

ENVIRONMENT VARIABLES (Alternative to secrets.toml):
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-public-key
KITE_API_KEY=your-kite-api-key
KITE_API_SECRET=your-kite-api-secret
KITE_REDIRECT_URI=your-redirect-uri
GEMINI_API_KEY=your-gemini-api-key
*/
