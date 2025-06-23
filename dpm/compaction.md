let's talk about
leveled compaction now hang on to your
hat
this is a little complicated level
compaction
probably is the most complicated
actually
it is the most complicated way to
compact data compaction in general is
not a very understood topic
and if you can get through this one all
the other compaction schemes are a lot
easier
but it is a pretty important process
having compaction
for certain workloads and having it
tailored to workloads
makes a lot of sense given the amount of
data that has to get moved around
and how much disk io has to be generated
let's start from the beginning level
Level Compaction
compaction
i'm going to show you this and how it
works
graphically and i think this is the best
way to understand level compaction
actually any compaction if i explain it
to you
you wouldn't be able to explain it back
if i show you i think it will make a lot
more sense
so how does it work well level
compaction the leveled part
actually is important because as an ss
table is written to disk
that data is sitting there it's
immutable however when that data is
written
now we have to employ some sort of
compaction strategy
with it level compaction looks at that
first
ss table that's written to disk as level
zero
the first file and as i write more data
another flush to disk that goes into
level zero
now what happens we just don't want
these things filling up the disk
the way they are compaction is an
important process so
if i want to compact using level
compaction what do i do now with these
two ss tables
well this is where the level comes in we
move it to the next level up
it gets promoted to level one how it
does that
similar like any other compaction
strategy it tries to group things
together
so in this case the 12s in that level
we're looking at the max ss table size
so this is what's going to help guide us
through our levels
so we're inside of that max ss table
size right now
great let's do another compaction boom
just went over
okay now what well we create a new file
so we do another compaction with the 52s
that goes over
we do the next one it's important to
note that level zero is really the
landing place
everything in level zero will get
promoted so it's right away
so as we compact everything's going
beyond the max ss table size
awesome okay level one is now ready to
go
however with the maxes table size comes
another
parameter and that is the maximum size
for level one
the max level one size is a multiplier
of the max
ss table size in this example we're
going to say 2
but in the real world it's 10 that's the
default so let's look at the example
again
if max l1 size is 2 times the max as
table size
that's how much we can store in level
one we're over
so now we need to trigger another
compaction so
we could take those two sizes those two
maxes table sizes
but i'm gonna have to create a level two
to fit more data in my disk
to satisfy the leveling this is where
that level work comes from we're
building more levels
so what we can do is just move that file
down to level two
do a quick size check up we're still
over so we're gonna have to move another
file down
awesome now everything seems to fit we
have everything leveled properly
everything's in the right place
great the levels are settled level two
max size is a multiplier
again two times level one so you can see
how it starts growing
in that way so if we look at our data
everything's fitting neatly in the
levels
we are totally fine now there's no more
compaction that needs to get done
but inevitably we're going to start
looking at all the sizing
but what's going to happen ss tables are
going to still be written to disk
and when that happens it promotes the
level 0 to level 1
immediately and the same process happens
again
and it just keeps going and we're going
to start compacting and compacting again
now if you look at what we have here we
just went over the sizing again
so here we go again we're going to be
compacting down into level two
max ss table size is too big we can pack
that one down
we create a new ss table now we have
everything in level two
level one is satisfied we're within the
boundaries for all the different sizing
great but like i said we're not gonna
stop we're gonna keep writing data into
the system
this whole thing happens over and over
and over again
it's very intensive so when we're
writing data to the system
it gets promoted into the right levels
and those levels start moving down from
level one to level two to level three
level four the good thing here is that
we know where that data is going to be
and as it gets promoted further it gets
written less
now as i mentioned we used a multiplier
of two
two times max as this table size the
real world
it's ten so it didn't look as good an
example
but you get what we're trying to do here
we're showing multipliers
10 is the default you can change that if
you need to in this case
it's probably best to leave it alone the
Configuration
configuration as you can see here in the
diagram
ss table max size that's 160 meg that's
the default
you could change that again that's one
of those things that you better not
change it unless you know exactly what
you're doing because you could really
mess things up quickly
and nobody wants to deal with a blown up
level compaction system
it will get really crazy fast because of
how the levels work
you can have things promoting really
quickly and that's just going to eat up
a lot of disk i o
so once we have these ss tables
exceeding these amounts
that's when we go we do another write
in the example that we had we had pretty
large partitions so things are getting
promoted pretty regularly
in the real world they don't get
promoted quite as fast unless you're
doing a lot of writes
the smaller your partitions the closer
Size
you're going to get to that actual 160
meg size
that 160 megs is just a marker if it
goes over that's fine we're trying to
keep the data close to each other if you
go and look at the actual file system
you'll see that there's different sizes
there's not just 160 meg chunks of data
in your disk so that's what's going on
there when you go look
level of compaction is best used for
reads and here's why
so if you think about what you're trying
Consistency
to do with this data you're trying to
get it to a level where it's stopped
so occasional rights meaning it doesn't
promote a lot
but the reads are very consistent it
knows exactly where that data is and
it's
very stable meaning that level one level
two level three by the time you get to
level three
it's not going to be written as much and
you know that that data is always going
to be there
so it caches nicely it's just really
good for reads
the idea is that we're trying to group
this data together very neatly
and succinctly level compaction does
that level compaction is not just a
cassandra topic it is a general topic in
distributed systems
especially with log structured storage
however cassandra's done a lot
with level compaction tailoring it to
specific data models
for cassandra as you can see from this
table things add up really quickly
you're adding a zero every single time
there are multipliers
so it can get pretty large by the time
you get to level five or six
is a lot of data in that level so
between level one level two and level
three
we have a certain amount of size which
is about ten percent
of that level four disk usage is also
something you need to consider with
level compaction
the general rule about compaction is you
General Rule
need a little more space
not everything so it isn't 50 more of
the disk space you have and the reason
being is because you can have
one level above overlap the level below
it needs to be able to compact
everything from that size down one level
so
a little more 11x is generally the
multiplier per level
if you need to compact from level 1 to
level two or level two to level three
you can predict how much you're going to
need for that so if you're at level two
then you're gonna need 1.6 terabytes of
space
left for level three if you're filling
up level three
you're going to need a 160 gigabytes
plus a little more
to compact down to level four in general
Advantages
the disk usage wastes a lot less of the
disk and that's a good thing
other compaction strategies aren't quite
as miserly with your disk
level compaction is really good for that
the other advantage is that obsolete
records
can disappear really quickly as they get
promoted they get removed
the downside though is it is very i o
intensive
you can see some of these aren't really
that good io intensity that's the
biggest problem with level compaction
Disadvantages
if you're doing a lot of writes you're
generating a lot of level zero files
which didn't get promoted the more
promotion means more big files have to
get moved around
if you're going from say level three to
level four that's a lot of data moving
around
so that's why high rights on a level
compaction system
is not a good idea you should consider
something like size tier or time window
in that case
but that disk intensity is really what
we're trying to avoid anyway because
cassandra does love to use disc io and
it's important for your reeds
you do not want compaction impacting
your reeds another problem that could
happen
is with all these levelings is that
eventually compaction will start falling
behind
level zero will get really far behind
and then you're in trouble
Safety Valves
one of the safety valves for that is
that level zero will then switch to
size to your compaction to bring that
file size down
it's a catch-up mechanism but what it
does is it eliminates the need to
promote
those ss tables to the next level in
that emergency state
and that is an emergency state it
shouldn't happen all the time it's there
to keep you from having all these files
pending
waiting to go to level one when really
your system is running out of disk io
well i hope you got through this okay
level compaction is definitely the
hardest one to understand
the different levels and how they work
somewhat complicated
it's not as bad as some people think
hopefully the graphics really helped you
visualize
what's going on with those files as
they're getting promoted
and just know some of the drawbacks the
good and the bad and level of compaction
could be your friend
