-- Telecom-RAG Supabase schema
-- Assumes BAAI/bge-small-en-v1.5 (384-dimensional embeddings).

create schema if not exists extensions;
create extension if not exists vector with schema extensions;

create table if not exists public.document_chunks (
    chunk_id text primary key,
    source_id text not null,
    source text not null,
    title text,
    page integer,
    section text,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding extensions.vector(384) not null,
    embedding_model text not null,
    chunking_version text not null,
    fts tsvector generated always as (
        to_tsvector(
            'english'::regconfig,
            coalesce(title, '') || ' ' ||
            coalesce(section, '') || ' ' ||
            content
        )
    ) stored,
    updated_at timestamptz not null default now()
);

create index if not exists document_chunks_embedding_hnsw_idx
    on public.document_chunks
    using hnsw (embedding extensions.vector_cosine_ops);

create index if not exists document_chunks_fts_idx
    on public.document_chunks using gin (fts);

create index if not exists document_chunks_config_idx
    on public.document_chunks (embedding_model, chunking_version);


create table if not exists public.kpi_observations (
    observation_id text primary key,
    "timestamp" timestamptz,
    orientation text,
    latitude double precision,
    longitude double precision,
    altitude double precision,
    lte_cell_id text,
    nr_cell_id text,
    lte_rsrp_dbm double precision,
    nr_rsrp_dbm double precision,
    lte_sinr_db double precision,
    nr_sinr_db double precision,
    nr_cqi double precision,
    nr_mcs double precision,
    nr_ri double precision,
    throughput_mbps double precision,
    source_sampling_interval_s double precision,
    anomaly_score double precision,
    anomaly_flag boolean,
    payload jsonb not null default '{}'::jsonb,
    updated_at timestamptz not null default now()
);

create index if not exists kpi_observations_timestamp_idx
    on public.kpi_observations ("timestamp");

create index if not exists kpi_observations_anomaly_idx
    on public.kpi_observations (anomaly_score desc nulls last);


-- Dense pgvector retrieval.
create or replace function public.match_document_chunks(
    p_query_embedding extensions.vector(384),
    p_match_count integer default 15,
    p_embedding_model text default 'BAAI/bge-small-en-v1.5',
    p_chunking_version text default 'section-aware-v2'
)
returns table (
    chunk_id text,
    source_id text,
    source text,
    title text,
    page integer,
    section text,
    content text,
    metadata jsonb,
    similarity double precision,
    retrieval_methods text
)
language sql
stable
security definer
set search_path = ''
as $$
    select
        dc.chunk_id,
        dc.source_id,
        dc.source,
        dc.title,
        dc.page,
        dc.section,
        dc.content,
        dc.metadata,
        (1 - (dc.embedding <=> p_query_embedding))::double precision as similarity,
        'dense'::text as retrieval_methods
    from public.document_chunks dc
    where dc.embedding_model = p_embedding_model
      and dc.chunking_version = p_chunking_version
    order by dc.embedding <=> p_query_embedding
    limit greatest(p_match_count, 1);
$$;


-- Hybrid full-text + pgvector retrieval using reciprocal-rank fusion.
create or replace function public.hybrid_search_document_chunks(
    p_query_text text,
    p_query_embedding extensions.vector(384),
    p_match_count integer default 20,
    p_rrf_k integer default 60,
    p_embedding_model text default 'BAAI/bge-small-en-v1.5',
    p_chunking_version text default 'section-aware-v2'
)
returns table (
    chunk_id text,
    source_id text,
    source text,
    title text,
    page integer,
    section text,
    content text,
    metadata jsonb,
    similarity double precision,
    text_rank real,
    rrf_score double precision,
    retrieval_methods text
)
language sql
stable
security definer
set search_path = ''
as $$
with params as (
    select websearch_to_tsquery('english'::regconfig, p_query_text) as tsq
),
semantic as (
    select
        dc.chunk_id,
        (1 - (dc.embedding <=> p_query_embedding))::double precision as similarity,
        row_number() over (order by dc.embedding <=> p_query_embedding) as semantic_rank
    from public.document_chunks dc
    where dc.embedding_model = p_embedding_model
      and dc.chunking_version = p_chunking_version
    order by dc.embedding <=> p_query_embedding
    limit greatest(p_match_count * 3, 30)
),
keyword as (
    select
        dc.chunk_id,
        ts_rank_cd(dc.fts, params.tsq)::real as text_rank,
        row_number() over (
            order by ts_rank_cd(dc.fts, params.tsq) desc
        ) as keyword_rank
    from public.document_chunks dc
    cross join params
    where dc.embedding_model = p_embedding_model
      and dc.chunking_version = p_chunking_version
      and dc.fts @@ params.tsq
    order by ts_rank_cd(dc.fts, params.tsq) desc
    limit greatest(p_match_count * 3, 30)
),
fused as (
    select
        coalesce(s.chunk_id, k.chunk_id) as chunk_id,
        s.similarity,
        k.text_rank,
        (
            case
                when s.semantic_rank is not null
                then 1.0 / (greatest(p_rrf_k, 1) + s.semantic_rank)
                else 0
            end
            +
            case
                when k.keyword_rank is not null
                then 1.0 / (greatest(p_rrf_k, 1) + k.keyword_rank)
                else 0
            end
        )::double precision as rrf_score,
        concat_ws(
            '+',
            case when s.semantic_rank is not null then 'dense' end,
            case when k.keyword_rank is not null then 'fts' end
        ) as retrieval_methods
    from semantic s
    full outer join keyword k on s.chunk_id = k.chunk_id
)
select
    dc.chunk_id,
    dc.source_id,
    dc.source,
    dc.title,
    dc.page,
    dc.section,
    dc.content,
    dc.metadata,
    f.similarity,
    f.text_rank,
    f.rrf_score,
    f.retrieval_methods
from fused f
join public.document_chunks dc on dc.chunk_id = f.chunk_id
order by f.rrf_score desc
limit greatest(p_match_count, 1);
$$;


create or replace function public.telecom_rag_status(
    p_embedding_model text default 'BAAI/bge-small-en-v1.5',
    p_chunking_version text default 'section-aware-v2'
)
returns jsonb
language sql
stable
security definer
set search_path = ''
as $$
    select jsonb_build_object(
        'document_chunks_total', (select count(*) from public.document_chunks),
        'document_chunks_current', (
            select count(*)
            from public.document_chunks
            where embedding_model = p_embedding_model
              and chunking_version = p_chunking_version
        ),
        'kpi_observations', (select count(*) from public.kpi_observations),
        'embedding_model', p_embedding_model,
        'chunking_version', p_chunking_version
    );
$$;


-- Runtime access: public dataset rows may be read, but document chunks are exposed
-- only through the two constrained search RPCs.
alter table public.document_chunks enable row level security;
alter table public.kpi_observations enable row level security;

drop policy if exists "public read kpi observations" on public.kpi_observations;
create policy "public read kpi observations"
    on public.kpi_observations
    for select
    to anon, authenticated
    using (true);

revoke all on table public.document_chunks from anon, authenticated;
grant select on table public.kpi_observations to anon, authenticated;

grant execute on function public.match_document_chunks(
    extensions.vector, integer, text, text
) to anon, authenticated;

grant execute on function public.hybrid_search_document_chunks(
    text, extensions.vector, integer, integer, text, text
) to anon, authenticated;

grant execute on function public.telecom_rag_status(text, text)
    to anon, authenticated;
